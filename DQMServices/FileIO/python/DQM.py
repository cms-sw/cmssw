#!/bin/env python3

from __future__ import print_function
from builtins import range
import ROOT as R
import os, re

class DQMReader(object):
    """
    Reader for  DQM IO and DQM root files.
    """

    #defined DQMIO types, index is important!
    DQMIO_TYPES = ["Ints","Floats","Strings",
         "TH1Fs","TH1Ss","TH1Ds",
         "TH2Fs", "TH2Ss", "TH2Ds",
         "TH3Fs", "TProfiles","TProfile2Ds", "kNIndicies"]

    def __init__(self, input_filename):
        self._root_file = R.TFile.Open(input_filename)

        ioTest = self._root_file.Get("Indices")
        if bool(ioTest):
            self.type = "DQMIO"
        else:
            self.type = "ROOT"

    def read_objects(self):
        if (self.type == "DQMIO"):
            return self.read_objects_dqmio()
        else:
            return self.read_objects_root()

    def read_objects_dqmio(self):
        indices = self._root_file.Get("Indices")

        for y in range(indices.GetEntries()):
            indices.GetEntry(y)
            # print indices.Run, indices.Lumi, indices.Type

            if indices.Type == 1000:
                # nothing is stored here
                # see https://github.com/cms-sw/cmssw/blob/8be445ac6fd9983d69156199d4d1fd3350f05d92/DQMServices/FwkIO/plugins/DQMRootOutputModule.cc#L437
                continue

            object_type = self.DQMIO_TYPES[indices.Type]
            t_tree = self._root_file.Get(object_type)

            for i in range(indices.FirstIndex, indices.LastIndex + 1):
                t_tree.GetEntry(i)

                fullname = str(t_tree.FullName)
                yield (fullname, t_tree.Value, )

    def read_objects_root(self):
        xml_re = re.compile(r"^<(.+)>(.+)=(.*)<\/\1>$")
        def parse_directory(di):
            directory = self._root_file.GetDirectory(di)
            for key in directory.GetListOfKeys():
                entry = key.GetName()
                rtype = key.GetClassName()
                fullpath = "%s/%s" % (di, entry)

                if (rtype == "TDirectoryFile"):
                    for k, v in parse_directory(fullpath):
                        yield (k, v, )
                else:
                    obj = self._root_file.Get(fullpath)
                    if obj:
                        yield (fullpath, obj, )
                    else:
                        # special case to parse the xml abomination
                        m = xml_re.search(entry)
                        if m:
                            name = m.group(1)
                            typecode = m.group(2)
                            value = m.group(3)

                            fp = "%s/%s" % (di, name)
                            yield (fp, value, )
                        else:
                            raise Exception("Invalid xml:" + entry)


        path_fix = re.compile(r"^\/Run \d+")
        for fullname, obj in parse_directory(""):
            f = fullname.replace("/DQMData", "")
            f = f.replace("/Run summary", "")
            f = path_fix.sub(r"", f)
            if f[0] == "/":
                f = f[1:]

            yield f, obj

    def close(self):
        self._root_file.Close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help = "Input DQMIO ROOT file")
    args = parser.parse_args()

    reader = DQMReader(args.input)

    for (fn, v) in reader.read_objects():
        if (hasattr(v, "ClassName")):
            print(fn, v.ClassName())
        else:
            print(fn, type(v))

    reader.close()
