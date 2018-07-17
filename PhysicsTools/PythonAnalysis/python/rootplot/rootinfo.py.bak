"""
Print information about objects in a ROOT file.
"""
from __future__ import absolute_import

from .version import __version__

from ROOT import Double
import copy
from . import argparse
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
            obj = in_tfile.Get(full_name)
            if not obj:
                continue
            simple_name = full_name[2:]
            print "%s" % simple_name,
            for arg in [x[2:] for x in sys.argv if x.startswith("--")]:
                if "classname" == arg:
                    print "%s" % classname,
                if obj.InheritsFrom('TH1'):
                    if "entries" == arg:
                        print " %i" % obj.GetEntries(),
                    if "contents" == arg:
                        if obj.InheritsFrom('TH2'):
                            # Print contents as they would look on the 2D graph
                            # Left to right, top to bottom.  Start in upper left corner.
                            for j in reversed(range(obj.GetNbinsY())):
                                print
                                print " %s" % ' '.join(
                                    [str(obj.GetBinContent(i+1, j+1)) for i in range(obj.GetNbinsX())]),
                        else:
                            print " %s" % ' '.join(
                                [str(obj.GetBinContent(i+1)) for i in range(obj.GetNbinsX())]),
                    if "errors" == arg:
                        if obj.InheritsFrom('TH2'):
                            for j in reversed(range(obj.GetNbinsY())):
                                print
                                print " %s" % ' '.join(
                                    [str(obj.GetBinError(i+1, j+1)) for i in range(obj.GetNbinsX())]),
                        else:
                            print " %s" % ' '.join(
                                [str(obj.GetBinError(i+1)) for i in range(obj.GetNbinsX())]),
                    if "bincenter" == arg:
                        print " %s" % ' '.join(
                            [str(obj.GetBinCenter(i+1)) for i in range(obj.GetNbinsX())]),
                    if "max" == arg:
                        print " %i" % obj.GetMaximum(),
                    if "min" == arg:
                        print " %i" % obj.GetMinimum(),
                    if "overflow" == arg:
                        print " %i" % obj.GetBinContent(obj.GetNbinsX()),
                    if "underflow" == arg:
                        print " %i" % obj.GetBinContent(0),
                if obj.InheritsFrom('TGraph'):
                    if "contents" == arg:
                        x, y = Double(0), Double(0)
                        xvals = []
                        yvals = []
                        for i in range(obj.GetN()):
                            obj.GetPoint(i, x, y)
                            xvals.append(copy.copy(x))
                            yvals.append(copy.copy(y))
                        for point in zip(xvals,yvals):
                            print " (%d, %d)" % point,
            print ""

def main():
    parser = argparse.ArgumentParser(description='Print information from an SC2 replay file.')
    parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                        help="Names of one or more root files")
    parser.add_argument('--bincenter', action="store_true", default=False,
                      help="Get Bin Centers from each bin in each histogram")
    parser.add_argument('--classname', action="store_true", default=False,
                      help="Get type from each object in root file")
    parser.add_argument('--contents', action="store_true", default=False,
                      help="Get Bin Contents from each bin in each histogram")
    parser.add_argument('--errors', action="store_true", default=False,
                      help="Get Bin Errors from each bin in each histogram")
    parser.add_argument('--entries', action="store_true", default=False,
                      help="Get Entries from each histogram")
    parser.add_argument('--max', action="store_true", default=False,
                      help="Get Maximum value from each histogram")
    parser.add_argument('--min', action="store_true", default=False,
                      help="Get Minimum value from each histogram")
    parser.add_argument('--name', default=None,
                      help="Get information only from object with matching name")
    parser.add_argument('--overflow', action="store_true", default=False,
                      help="Get value of overflow bin from each histogram")
    parser.add_argument('--underflow', action="store_true", default=False,
                      help="Get value of underflow bin from each histogram")
    arguments = parser.parse_args()
    for arg in arguments.filenames:
        if arg[-5:] != ".root":
            raise TypeError("Arguments must include root file names")
    filenames_from_interface = [x for x in arguments.filenames if x[-5:] == ".root"]
    if len(filenames_from_interface) == 0:
        parser.print_help()
        sys.exit(0)
    for filename in filenames_from_interface:
        if not os.path.exists(filename):
            print "%s does not exist." % filename
            sys.exit(0)
        tfile = TFile(filename, "read")
        try:
            recurse_thru_file(tfile, arguments)
        except IOError as e:
            if e.errno != 32:
                raise

if __name__ == '__main__':
    main()
