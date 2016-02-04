#!/usr/bin/env python
'''
Get information about contents of a ROOT file.
'''

from version import __version__

import optparse
import sys
import os

#### Load classes from ROOT, ensuring it doesn't intercept -h or --help
saved_argv = sys.argv[:]
sys.argv = [sys.argv[0], '-b']
from ROOT import TFile, TH1, TDirectory, gDirectory
sys.argv = saved_argv

def recurse_thru_file(in_tfile, options, full_path='/'):
    '''Recursive function to find all contents in a given ROOT file'''
    keys = in_tfile.GetDirectory(full_path).GetListOfKeys()
    for key in keys:
        name = key.GetName()
        classname = key.GetClassName()
        if 'TDirectory' in classname:
            gDirectory.cd(name)
            recurse_thru_file(in_tfile, options, '/'.join([full_path,name]))
            gDirectory.cd("..")
        else:
            if options.name and name != options.name: continue
            full_name = '/'.join([full_path,name])
            object = in_tfile.Get(full_name)
            simple_name = full_name[2:]
            print "%s" % simple_name,
            for arg in [x[2:] for x in sys.argv if x.startswith("--")]:
                if "classname" == arg:
                    print "%s" % classname,
                if object.InheritsFrom('TH1'):
                    if "entries" == arg:
                        print " %i" % object.GetEntries(),
                    if "contents" == arg:
                        print " %s" % ' '.join(
                            [str(object.GetBinContent(i+1)) for i in range(object.GetNbinsX())]),
                    if "bincenter" == arg:
                        print " %s" % ' '.join(
                            [str(object.GetBinCenter(i+1)) for i in range(object.GetNbinsX())]),
                    if "max" == arg:
                        print " %i" % object.GetMaximum(),
                    if "min" == arg:
                        print " %i" % object.GetMinimum(),
                    if "overflow" == arg:
                        print " %i" % object.GetBinContent(object.GetNbinsX()),
                    if "underflow" == arg:
                        print " %i" % object.GetBinContent(0),
            print ""

def main():
    usage=("Print information about histograms in a root file")
    parser = optparse.OptionParser(usage=usage,
                                   version='%s %s' % ('%prog', __version__))
    parser.add_option('--bincenter', action="store_true", default=False,
                      help="Get Bin Centers from each bin in each histogram")
    parser.add_option('--classname', action="store_true", default=False,
                      help="Get type from each object in root file")
    parser.add_option('--contents', action="store_true", default=False,
                      help="Get Bin Contents from each bin in each histogram")
    parser.add_option('--entries', action="store_true", default=False,
                      help="Get Entries from each histogram")
    parser.add_option('--max', action="store_true", default=False,
                      help="Get Maximum value from each histogram")
    parser.add_option('--min', action="store_true", default=False,
                      help="Get Minimum value from each histogram")
    parser.add_option('--name', default=None,
                      help="Get information only from object with matching name")
    parser.add_option('--overflow', action="store_true", default=False,
                      help="Get Minimum value from each histogram")
    parser.add_option('--underflow', action="store_true", default=False,
                      help="Get Minimum value from each histogram")
    options, arguments = parser.parse_args()
    for arg in arguments:
        if arg[-5:] != ".root":
            raise TypeError("Arguments must include root file names")
    filenames_from_interface = [x for x in arguments if x[-5:] == ".root"]
    if len(filenames_from_interface) == 0:
        parser.print_help()
        sys.exit(0)
    for filename in filenames_from_interface:
        if not os.path.exists(filename):
            print "%s does not exist." % filename
            sys.exit(0)
        tfile = TFile(filename, "read")
        recurse_thru_file(tfile, options)

if __name__ == '__main__':
    main()
