#!/usr/bin/env python
from __future__ import print_function
import re
import json
import ROOT
import sqlite3
import argparse

parser = argparse.ArgumentParser(description="Convert arbitrary ROOT file to SQLite database, mapping TTrees to tables and converting TObjects to JSON.")

parser.add_argument('inputfile', help='ROOT file to read')
parser.add_argument('-o', '--output', help='SQLite file to write', default='root.sqlite')
args = parser.parse_args()

f = ROOT.TFile.Open(args.inputfile)
db = sqlite3.connect(args.output)

basic_objects = {}

inf = re.compile("([- \[])inf([,}\]])")
nan = re.compile("([- \[])nan([,}\]])")

def tosqlite(x):
    if isinstance(x, ROOT.string):
        try:
            return unicode(x.data())
        except:
            return buffer(x.data())
    if isinstance(x, int):
          return x
    if isinstance(x, float):
        return x
    if isinstance(x, long):
        return x
    else:
        try: 
            rootobj = unicode(ROOT.TBufferJSON.ConvertToJSON(x))
            # turns out ROOT does not generate valid JSON for NaN/inf
            clean = nan.sub('\\g<1>0\\g<2>', inf.sub('\\g<1>1e38\\g<2>', rootobj))
            obj = json.loads(clean)
            jsonobj = json.dumps(obj, allow_nan=False)
            return jsonobj
        except Exception as e:
            return json.dumps({"root2sqlite_error": e.__repr__(), "root2sqlite_object": x.__repr__()})

def columnescape(s):
    # add whatever is not a valid column name here
    SQLKWDs = ["index"]
    if s.lower() in SQLKWDs:
        return s + "_"
    else:
        return s

def treetotable(ttree, name):
    name = name.replace("/", "_")
    branches = [b.GetName() for b in ttree.GetListOfBranches()]
    colnames = ", ".join(columnescape(b) for b in branches)
    create =  "CREATE TABLE %s(%s);"  % (name, colnames)
    print(create)
    db.execute(create)
    data = []
    for i in range(ttree.GetEntries()):
        ttree.GetEntry(i)
        vals = tuple([tosqlite(getattr(ttree, b)) for b in branches])
        data.append(vals)
    insert = "INSERT INTO %s(%s) VALUES (%s);" % (name, colnames, ",".join(["?"] * len(branches)))
    print(insert)
    db.executemany(insert, data)

def read_objects_root(rootfile):
    xml_re = re.compile(r"^<(.+)>(.+)=(.+)<\/\1>$")
    def parse_directory(di):
        directory = rootfile.GetDirectory(di)
        for key in directory.GetListOfKeys():
            entry = key.GetName()
            rtype = key.GetClassName()
            fullpath = "%s/%s" % (di, entry) if di != "" else entry
            if (rtype == "TDirectoryFile"):
                for k, v, t in parse_directory(fullpath):
                    yield (k, v, t)
            else:
                obj = rootfile.Get(fullpath)
                if obj:
                    yield (fullpath, obj, rtype)
                else:
                    # special case to parse the xml abomination
                    m = xml_re.search(entry)
                    if m:
                        name = m.group(1)
                        typecode = m.group(2)
                        value = m.group(3)
                        fp = "%s/%s" % (di, name)
                        yield (fp, value, rtype)
                    else:
                        raise Exception("Invalid xml:" + entry)
    path_fix = re.compile(r"^\/Run \d+")
    for fullname, obj, rtype in parse_directory(""):
        yield fullname, obj, rtype

def save_keyvalue(dictionary, name):
    name = name.replace("/", "_")
    create =  "CREATE TABLE %s(key, value);"  % name
    print(create)
    db.execute(create)
    data = []
    for k, v in dictionary.iteritems():
        vals = (unicode(k), tosqlite(v))
        data.append(vals)
    insert = "INSERT INTO %s(key, value) VALUES (?,?);" % name
    print(insert)
    db.executemany(insert, data)
    db.commit()


for name, obj, rtype in read_objects_root(f):
  if rtype == "TTree":
    treetotable(obj, name)
  else:
    basic_objects[name] = obj

save_keyvalue(basic_objects, "TDirectory")
