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
## @param fin is the ROOT input file (the TFile, not the file name)
#
def findTopDir(fin):
    """tries to find a top directory for the DQM histograms. Note
    that the run number seems to be always 1 for MC but differs
    for data. If there is more than one top directory, this function
    prints an error message on stderr and exits (maybe this should
    be made more flexible in the future in order to allow DQM histogramming
    of data of multiple runs).

    Returns None if no full path could be found.

    """

    import re

    # an path looks like:
    # "DQMData/Run <run>/HLT/Run summary/HLTEgammaValidation"

    theDir = fin.Get("DQMData")

    if theDir == None:
        return None

    # now look for directories of the form 'Run %d'

    runSubdirName = None

    for subdirName in [ x.GetName() for x in theDir.GetListOfKeys() ]:

        if re.match("Run \d+$", subdirName):
            if runSubdirName != None:
                # more than one run found
                print >> sys.stderr,"more than one run found in the DQM file, this is currently not supported"
                sys.exit(1)

            runSubdirName = subdirName


    # check that we have at least (exactly) one directory
    if runSubdirName == None:
        return None

    # get the rest
    return theDir.Get(runSubdirName + "/HLT/Run summary/HLTEgammaValidation")
 

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

parser.add_option("--ignore-empty",
                  dest="ignore_empty_paths",
                  action='store_true',
                  default = False,
                  help="Print only information about non-empty paths (i.e. those with at least one entry in the total_eff histogram).",
                  )

parser.add_option("--ignore-zero-eff",
                  dest="ignore_zero_efficiency",
                  action='store_true',
                  default = False,
                  help="Print only information about paths which have at least one entry in the bin of the last module in the overview histogram. Note that this also excludes those paths excluded by --ignore-empty .",
                  )

parser.add_option("--no-split-names",
                  dest="split_names",
                  action='store_false',
                  default = True,
                  help="Do not split module names.",
                  )


(options, ARGV) = parser.parse_args()

if len(ARGV) != 1:
    parser.print_help()
    sys.exit(1)

#----------------------------------------
# open the ROOT file
#----------------------------------------

fin = ROOT.TFile.Open(ARGV[0])

top_dir = findTopDir(fin)

if top_dir == None:
    print >> sys.stderr,"could not find a top directory inside root file"
    print >> sys.stderr,"A typical top directory for MC is 'DQMData/Run 1/HLT/Run summary/HLTEgammaValidation'"
    print >> sys.stderr
    print >> sys.stderr,"Exiting"
    sys.exit(1)


#--------------------
# determine the length of the longest path name (for nice printout)
#--------------------

maxPathNameLen = None
allPathNames = []

for path_key in top_dir.GetListOfKeys():

    pathName = path_key.GetName()

    if len(options.selected_paths) != 0 and not path_name in options.selected_paths:
        continue

    # further checks which are done in the next
    # loop are not repeated here.
    # so we might get a maximum number of characters
    # which is slightly too high (but the code here
    # is more readable)

    allPathNames.append(pathName)

    maxPathNameLen = max(maxPathNameLen, len(pathName))

#--------------------

for path_name in allPathNames:

    path_dir = top_dir.Get(path_name)

    # just select directories (there are also other
    # objects in the top directory)
    if not isinstance(path_dir,ROOT.TDirectoryFile):
        continue

    # find modules in order from total_eff_MC_matched histogram
    total_eff_histo = path_dir.Get("total_eff_MC_matched")

    if total_eff_histo == None:
        # try with data:
        total_eff_histo = path_dir.Get("total_eff_RECO_matched")

    # subtract 2 for 'Total' and 'Gen' bins
    num_modules = total_eff_histo.GetNbinsX() - 2

    total = total_eff_histo.GetBinContent(num_modules)
    num_gen_events = total_eff_histo.GetBinContent(num_modules + 2)

    if num_gen_events == 0 and options.ignore_empty_paths:
        continue

    # check whether at least one event passed all modules
    if options.ignore_zero_efficiency:
        # get number of entries in last module

        last_module_index = num_modules - 1
        
        last_module_accepted_events = total_eff_histo.GetBinContent(last_module_index+1)

        if last_module_accepted_events < 1:
            continue

    
    #--------------------

    if not options.summary_mode:
        print "----------------------------------------"

    print ("PATH: %-" + str(maxPathNameLen) + "s") % path_name,

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

        if options.split_names:
            module_name = splitAtCapitalization(module_name)

        events = total_eff_histo.GetBinContent(i+1)

        


        print "  %-90s: %5d events" % (module_name, events),

        if previous_module_output > 0:
            eff = 100 * events / float(previous_module_output)
            print "(%5.1f%% eff.)" % (eff),
            if eff > 100.:
                if module_name.find("Unseeded") >= 0:
                    print ">100% Unseeded Filter",
                else:
                    print "ERROR",

        print
                                     

        previous_module_output = events


    print 
    

    


