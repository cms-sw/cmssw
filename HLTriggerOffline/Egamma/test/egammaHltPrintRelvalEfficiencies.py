#!/usr/bin/env python

import sys, os

try:
    import ROOT
except ImportError:
    print >> sys.stderr
    print >> sys.stderr,"  Error importing the ROOT python module"
    print >> sys.stderr,"  Try e.g. initializing a CMSSW environment"
    print >> sys.stderr,"  prior to starting this script"
    print >> sys.stderr
    sys.exit(1)

#----------------------------------------------------------------------

def splitAtCapitalization(text):
    """ splits a string before capital letters. Useful to make
    identifiers which consist of capitalized words easier to read

    We should actually find a smarter algorithm in order to avoid
    splitting things like HLT or LW. 

    """

    retval = ''

    for ch in text:
        if ch.isupper() and len(retval) > 0:
            retval += ' '

        retval += ch

    return retval
    

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
from optparse import OptionParser

parser = OptionParser("""

  usage: %prog [options] root_file

    given the output of the E/gamma HLT validation histogramming module,
    (DQM output) prints some information about path and module efficiencies.

    Useful for determining which paths actually have some meaningful
    results in the file and which ones not.
""")

parser.add_option("--summary",
                  dest="summary_mode",
                  default = False,
                  action="store_true",
                  help="print path efficiencies only, nothing about modules",
                  )

parser.add_option("--path",
                  dest="selected_paths",
                  default = [],
                  action="append",
                  help="restrict printout to specific path. "+ 
                       "This option can be given more than once to select several paths.",
                  )
(options, ARGV) = parser.parse_args()

if len(ARGV) != 1:
    parser.print_help()
    sys.exit(1)

#----------------------------------------
# open the ROOT file

fin = ROOT.TFile.Open(ARGV[0])

top_path = "DQMData/Run 1/HLT/Run summary/HLTEgammaValidation"
top_dir = fin.Get(top_path)

if top_dir == None:
    print >> sys.stderr,"could not find top directory " + top_path + " inside root file"
    sys.exit(1)


for path_key in top_dir.GetListOfKeys():

    path_name = path_key.GetName()

    if len(options.selected_paths) != 0 and not path_name in options.selected_paths:
        continue

    path_dir = top_dir.Get(path_name)

    # just select directories (there are also other
    # objects in the top directory)
    if not isinstance(path_dir,ROOT.TDirectoryFile):
        continue

    # find modules in order from total_eff_MC_matched histogram
    total_eff_histo = path_dir.Get("total_eff_MC_matched")

    # subtract 2 for 'Total' and 'Gen' bins
    num_modules = total_eff_histo.GetNbinsX() - 2

    total = total_eff_histo.GetBinContent(num_modules)
    num_gen_events = total_eff_histo.GetBinContent(num_modules + 1)

    if not options.summary_mode:
        print "----------------------------------------"

    print "PATH: %-60s" % path_name,

    if num_gen_events > 0:
        print "(%.1f%% eff.)" % (100 * total / float(num_gen_events)),

    elif options.summary_mode:
        print "(no entries)",

    print

    if not options.summary_mode:
        print "----------------------------------------"

        print "  %-80s: %5d events" % ('generated', num_gen_events)

    if options.summary_mode:
        continue

    previous_module_output = num_gen_events

    print

    for i in range(num_modules):

        module_name = total_eff_histo.GetXaxis().GetBinLabel(i+1)

        events = total_eff_histo.GetBinContent(i+1)

        print "  %-90s: %5d events" % (splitAtCapitalization(module_name), events),

        if previous_module_output > 0:
            print "(%5.1f%% eff.)" % (100 * events / float(previous_module_output)),

        print
                                     

        previous_module_output = events


    print 
    

    


