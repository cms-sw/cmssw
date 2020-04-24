#!/usr/bin/env python

import sys, os.path, json
from collections import defaultdict
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)


class FileData:
    def __init__(self,data):
        self._json = data
        for k,v in data.iteritems():
            setattr(self,k,v)
        self.Events = self.trees["Events"]
        self.nevents = self.Events["entries"]

class Branch:
    def __init__(self, tree, branch):
        self.tree = tree
        self.branch = branch
        self.name = branch.GetName()
        self.doc = branch.GetTitle()
        self.tot  = branch.GetZipBytes()/1024.0
        self.entries = None; 
        self.single = True
        self.kind   = "Unknown"
        if branch.GetNleaves() != 1:
            sys.stderr.write("Cannot parse branch '%s' in tree %s (%d leaves)\n", tree.GetName(), branch.GetName(), branch.GetNleaves())
            return
        self.leaf = branch.FindLeaf(branch.GetName())
        if not self.leaf:
            sys.stderr.write("Cannot parse branch '%s' in tree %s (no leaf)\n", tree.GetName(), branch.GetName())
            return
        self.kind = self.leaf.GetTypeName()
        if "Idx" in self.name:
                self.kind+="(index to %s)"%((self.name[self.name.find("_")+1:self.name.find("Idx")]).title())
        if self.leaf.GetLen() == 0 and self.leaf.GetLeafCount() != None:
            self.single = False
            self.counter = self.leaf.GetLeafCount().GetName()
    def toJSON(self):
        return ( self.name, dict(name = self.name, doc = self.doc, tot=self.tot, entries=self.entries, single=self.single, kind=self.kind, counter = getattr(self,'counter','')) )

class BranchGroup:
    def __init__(self, name):
        self.name = name
        self.tot  = 0
        self.entries = None; 
        self.subs = []
        self.kind   = None
        self.doc    = ''
    def append(self, sub):
        self.subs.append(sub)
        self.tot += sub.tot
        if not self.doc: self.doc = sub.doc
    def getKind(self):
        if self.kind: return self.kind
        if len(self.subs) == 1:
            if self.subs[0].single: self.kind = "Variable"
            else: 
                self.kind = "Vector"
                self.counter = self.subs[0].counter
        else:
            allsingles, commonCounter = True, True
            counter = None
            for s in self.subs:
                if not s.single:
                    allsingles = False
                    if counter == None: counter = s.counter
                    elif counter != s.counter:
                        commonCounter = False
            if allsingles:
                self.kind = "Singleton"
            elif commonCounter:
                self.kind = "Collection"
                self.counter = counter
            else:
                self.kind = "ItsComplicated"
        return self.kind
    def toJSON(self):
        return (self.name, dict(name = self.name, doc = self.doc, kind = self.kind, tot = self.tot, entries = self.entries, subs = [s.name for s in self.subs]))


def inspectRootFile(infile):
    if not os.path.isfile(infile): raise RuntimeError
    filesize = os.path.getsize(infile)/1024.0
    tfile = ROOT.TFile.Open(infile)
    trees = {}
    for treeName in "Events", "Runs", "Lumis":
        toplevelDoc={}
        tree = tfile.Get(treeName)
        entries = tree.GetEntries()
        trees[treeName] = tree
        branchList = tree.GetListOfBranches()
        allbranches = [ Branch(tree,branchList.At(i)) for i in xrange(branchList.GetSize()) ]
        branchmap = dict((b.name,b) for b in allbranches)
        branchgroups = {}
        # make list of counters and countees
        counters = defaultdict(list)
        for b in allbranches:
            if not b.single:
                counters[b.counter].append(b.name)
            else:
                b.entries = entries
        c1 = ROOT.TCanvas("c1","c1")
        for counter,countees in counters.iteritems():
            n = tree.Draw(counter+">>htemp")
            if n != 0:
                htemp = ROOT.gROOT.FindObject("htemp")
                n = htemp.GetEntries() * htemp.GetMean()
                htemp.Delete()
            branchmap[counter].entries = entries
            for c in countees:
                br = branchmap[c] 
                br.entries = n
        # now we start to create branch groups
        for b in allbranches:
            if b.name in counters:
                 if len(b.doc) > 0:
                     toplevelDoc[b.name[1:]]=b.doc
                 continue # skip counters
            if "_" in b.name:
                head, tail = b.name.split("_",1)
            else:
                head = b.name
                toplevelDoc[b.name]=b.doc
            if head not in branchgroups:
                branchgroups[head] = BranchGroup(head)
            branchgroups[head].append(b)
        for bg in branchgroups.itervalues():
            if bg.name in toplevelDoc:
                bg.doc = toplevelDoc[bg.name]
            kind = bg.getKind()
            bg.entries = bg.subs[0].entries
            if kind == "Vector" or kind == "Collection":
                bg.append(branchmap[bg.counter])
            elif kind == "ItsComplicated":
                for counter in set(s.counter for s in bg.subs if not s.single):
                    bg.append(branchmap[counter])
        allsize_c = sum(b.tot for b in allbranches)
        allsize = sum(b.tot for b in branchgroups.itervalues())
        if abs(allsize_c - allsize) > 1e-6*(allsize_c+allsize):
            sys.stderr.write("Total size mismatch for tree %s: %10.4f kb vs %10.4f kb\n" % (treeName, allsize, allsize_c))
        trees[treeName] =  dict(
                entries = entries,
                allsize = allsize,
                branches = dict(b.toJSON() for b in allbranches),
                branchgroups = dict(bg.toJSON() for bg in branchgroups.itervalues()),
            )
        c1.Close()
        break # only Event tree for now
    tfile.Close()
    return dict(filename = os.path.basename(infile), filesize = filesize, trees = trees)

def makeSurvey(treeName, treeData):
    allsize = treeData['allsize']
    entries = treeData['entries']
    survey = list(treeData['branchgroups'].itervalues())
    survey.sort(key = lambda bg : - bg['tot'])
    scriptdata = []
    runningtotal = 0
    unit = treeName[:-1].lower()
    for s in survey:
        if s['tot'] < 0.01*allsize:
            tag = "<b>Others</b><br/>Size: %.0f b/%s (%.1f%%)" % ((allsize - runningtotal)/entries*1024, unit, (allsize-runningtotal)/allsize*100)
            scriptdata.append( "{ 'label':'others', 'tag':'top', 'size':%s, 'tip':'%s' }" % ((allsize-runningtotal)/entries, tag) )
            break
        else:
            tag = "<b><a href=\"#%s\">%s</a></b><br/>" % (s['name'],s['name']);
            tag += "Size: %.0f b/%s (%.1f%%)" % (s['tot']/entries*1024, unit, s['tot']/allsize*100);
            if (s['kind'] in ("Vector","Collection")) and s['entries'] > 0:
                tag += "<br/>Items/%s:  %.1f, %.0f b/item" %(unit, float(s['entries'])/entries, s['tot']/s['entries']*1024);
            scriptdata.append( "{ 'label':'%s', 'tag':'%s', 'size':%s, 'tip':'%s' }" % ( s['name'], s['name'], s['tot']/entries, tag) )
        runningtotal += s['tot']
    return (survey, "\n,\t".join(scriptdata))
  
def writeSizeReport(fileData, stream):
    filename = fileData.filename
    filesize = fileData.filesize
    events = fileData.nevents
    survey, scriptdata = makeSurvey("Events", filedata.Events)
    title = "%s (%.3f Mb, %d events, %.2f kb/event)" % (filename, filesize/1024.0, events, filesize/events)
    stream.write("""
    <html>
    <head>
        <title>{title}</title>
        <link rel="stylesheet" type="text/css" href="http://gpetrucc.web.cern.ch/gpetrucc/micro/patsize.css" />
        <script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.core.js"></script>
        <script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.pie.js"></script>
        <script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.dynamic.js"></script>
        <script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.tooltips.js"></script>
        <script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.key.js"></script>
    </head>
    <body>
    <a name="top" id="top"><h1>{title}</h1></a>
    <canvas id="mainCanvas" width="800" height="300">[No canvas support]</canvas>
    <script type="text/javascript">
    var data = [ {scriptdata} ];
    """.format(title = title, scriptdata = scriptdata)); ## must split into two write calls, since the javascript confounds str.format
    stream.write("""
    window.onload = function() {
        values = [];
        labels = [];
        keys   = [];
        tips   = [];
        for (var i = 0; i < data.length; i++) {
            values.push( data[i].size );
            labels.push( data[i].label );
            keys.push( data[i].label );
            tips.push( data[i].tip );
        }
        var chart = new RGraph.Pie("mainCanvas", values)
                    .Set('exploded', 7)
                    .Set('tooltips', tips)
                    .Set('tooltips.event', 'onmousemove')
                    .Set('key', labels)
                    .Set('key.position.graph.boxed', false)
                    .Draw();
    }
    </script>
    <h1>Event data</h1>
    <table>
    """); 
    stream.write("<tr class='header'><th>" + "</th><th>".join([ "collection", "kind", "vars", "items/evt", "kb/evt", "b/item", "plot", "%" ]) + "</th><th colspan=\"2\">cumulative %</th></tr>\n");
    grandtotal = filedata.Events['allsize']; runningtotal = 0
    for s in survey:
        stream.write("<th title=\"%s\"><a href='#%s'>%s</a></th><td style='text-align : left;'>%s</td><td>%d</td>" % (s['doc'],s['name'],s['name'],s['kind'].lower(),len(s['subs'])))
        stream.write("<td>%.2f</td><td>%.3f</td><td>%.1f</td>" % (s['entries']/events, s['tot']/events, s['tot']/s['entries']*1024 if s['entries'] else 0))
        stream.write("<td class=\"img\"><img src='http://gpetrucc.web.cern.ch/gpetrucc/micro/blue-dot.gif' width='%d' height='%d' /></td>" % (s['tot']/grandtotal*200,10))
        stream.write("<td>%.1f%%</td>" % ( s['tot']/grandtotal * 100.0))
        stream.write("<td>%.1f%%</td>" % ( (runningtotal+s['tot'])/grandtotal * 100.0))
        stream.write("<td>%.1f%%</td>" % ( (grandtotal-runningtotal)/grandtotal * 100.0))
        stream.write("</tr>\n")
        runningtotal += s['tot'];

    # all known data
    stream.write("<th>All Event data</th>")
    stream.write("<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><b>%.2f</b></td><td>&nbsp;</td>"  % (grandtotal/events))
    stream.write("<td class=\"img\"><img src=\"http://gpetrucc.web.cern.ch/gpetrucc/micro/green-dot.gif\" width='%d' height='10' />" % ( grandtotal/filesize*100.0))
    stream.write("</td><td>%.1f%%<sup>a</sup></td>" % (grandtotal/filesize*100.0))
    stream.write("</tr>\n")

    # other, unknown overhead
    stream.write("<th>Non per-event data or overhead</th>")
    stream.write("<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>%.2f</td><td>&nbsp;</td>" % ( (filesize-grandtotal)/events))
    stream.write("<td class=\"img\"><img src='http://gpetrucc.web.cern.ch/gpetrucc/micro/red-dot.gif' width='%d' height='%d' /></td>" % ( (filesize-grandtotal)/filesize * 100, 10 ))
    stream.write("<td>%.1f%%<sup>a</sup></td>" % ( (filesize-grandtotal)/filesize * 100.0 ))
    stream.write("</tr>\n")

    # all file
    stream.write("<th>File size</th>")
    stream.write("<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><b>%.2f</b></td><td>&nbsp;</td>" % (filesize/events))
    stream.write("<td>&nbsp;</td><td>&nbsp;</td></tr>\n")

    stream.write("""
    </table>
    Note: size percentages of individual event products are relative to the total size of Event data only.<br />
    Percentages with <sup>a</sup> are instead relative to the full file size.
    <h1>Events detail</h1>
    """)
    for s in sorted(survey, key = lambda s : s['name']):
        stream.write("<h2><a name='%s' id='%s'>%s</a> (%.1f items/evt, %.3f kb/evt) <sup><a href=\"#top\">[back to top]</a></sup></h2>" % (s['name'], s['name'], s['name'], s['entries']/events, s['tot']/events))
        stream.write("<table>\n")
        stream.write("<tr class='header'><th>" + "</th><th>".join( [ "branch", "kind", "b/event", "b/item", "plot", "%" ]) + "</th></tr>\n")
        subs = [ fileData.Events['branches'][b] for b in s['subs'] ]
        for b in sorted(subs, key = lambda s : - s['tot']):
            stream.write("<th title=\"%s\">%s</th><td style='text-align : left;'>%s</td><td>%.1f</td><td>%.1f</td>" % (b['doc'],b['name'], b['kind'], b['tot']/events*1024, b['tot']/s['entries']*1024 if s['entries'] else 0))
            stream.write("<td class=\"img\"><img src='http://gpetrucc.web.cern.ch/gpetrucc/micro/blue-dot.gif' width='%d' height='%d' /></td>" % ( b['tot']/s['tot']*200, 10 ))
            stream.write("<td>%.1f%%</td>" % (b['tot']/s['tot'] * 100.0))
            stream.write("</tr>\n")
        stream.write("</table>\n")
    stream.write("""
    </body></html>
    """)

def writeDocReport(fileData, stream):
    stream.write( """
    <html>
    <head>
        <title>Documentation for {filename} </title>
        <link rel="stylesheet" type="text/css" href="http://gpetrucc.web.cern.ch/gpetrucc/micro/patsize.css" />
    </head>
    <body>
    <h1>Content</h1>
    <table>
    """.format(filename=fileData.filename))
    stream.write( "<tr class='header'><th>Collection</th><th>Description</th></tr>\n" )
    groups = fileData.Events['branchgroups'].values()
    groups.sort(key = lambda s : s['name'])
    for s in groups:
        stream.write( "<th><a href='#%s'>%s</a></th><td style='text-align : left;'>%s</td></tr>\n" % (s['name'],s['name'],s['doc']) )
    stream.write( "</table>\n\n<h1>Events detail</h1>\n" )
    for s in groups:
        stream.write( "<h2><a name='%s' id='%s'>%s</a> <sup><a href=\"#top\">[back to top]</a></sup></h2>\n" % (s['name'], s['name'], s['name']) )
        stream.write( "<table>\n" )
        stream.write( "<tr class='header'><th>Object property</th><th>Type</th><th>Description</th></tr>\n" )
        subs = [ fileData.Events['branches'][b] for b in s['subs'] ]
        for b in sorted(subs, key = lambda s : s['name']):
            stream.write( "<th>%s</th><td style='text-align : left;'>%s</td><td style='text-align : left;'>%s</td>" % (b['name'], b['kind'], b['doc']) )
            stream.write( "</tr>\n" )
        stream.write( "</table>\n" )
    stream.write( """
    </body></html>
    """ )

def _maybeOpen(filename):
    return open(filename, 'w') if filename != "-" else sys.stdout

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] inputFile")
    parser.add_option("-j", "--json", dest="json", type="string", default=None, help="Write out json file")
    parser.add_option("-d", "--doc", dest="doc", type="string", default=None, help="Write out html doc")
    parser.add_option("-s", "--size", dest="size", type="string", default=None, help="Write out html size report")
    (options, args) = parser.parse_args()
    if len(args) != 1: raise RuntimeError("Please specify one input file")

    if args[0].endswith(".root"):
        filedata = FileData(inspectRootFile(args[0]))
    elif args[0].endswith(".json"):
        filedata = FileData(json.load(open(args[0],'r')))
    else: raise RuntimeError("Input file %s is not a root or json file" % args[0])
    
    if options.json:
        json.dump(filedata._json, _maybeOpen(options.json), indent=4)
        sys.stderr.write("JSON output saved to %s\n" % options.json)
    if options.doc:
        writeDocReport(filedata, _maybeOpen(options.doc))
        sys.stderr.write("HTML documentation saved to %s\n" % options.doc)
    if options.size:
        writeSizeReport(filedata, _maybeOpen(options.size))
        sys.stderr.write("HTML size report saved to %s\n" % options.size)
