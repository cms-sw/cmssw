#!/usr/bin/env python

import sys, os, urllib


base_url = "https://cmsweb.cern.ch/dqm/offline/start"

# maps from type to subdirectory


knownDatasets = {
    "photonJet" : { "dqmdir" : "Photon Summary" },
    
    "zee" : { "dqmdir" : "Zee Preselection", },

    "wen" : { "dqmdir" : "Wenu Preselection" },
} 

#----------------------------------------------------------------------

def fixDataSetTier(dataset):
    """ makes sure that the data tier of the given data set is
    GEN-SIM-RECO (which is what seems to be used in DQM).

    Also supports datasets without any data tier.
    """

    if dataset.startswith('/'):
        dataset = dataset[1:]

    if dataset.endswith('/'):
        dataset = dataset[:-1]
    
    parts = dataset.split("/")

    if len(parts) < 2 or len(parts) > 3:
        raise Exception("expected at two or three parts (found " + str(len(parts)) + ") separated by '/' for dataset '" + dataset +"'")

    if len(parts) == 2:
        # data tier missing
        parts.append("GEN-SIM-RECO")
    else:
        # replace the last part
        parts[-1] = "GEN-SIM-RECO"

    return "/" + "/".join(parts)


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
from optparse import OptionParser

parser = OptionParser("""

  usage: %prog [options] reference_dataset [ new_dataset ]


  If only one dataset is given, links to the corresponding relval
  plots in the offline DQM are printed.

  If two datasets are given, links to overlay plots are printed.

  Note that the datasets used in DQM seem to have the data tier
  (third part) GEN-SIM-RECO. This program will replace any
  existing data tier by this value. You may also specify
  a dataset without data tier (e.g. /RelValWE/CMSSW_3_8_0-START38_V7-v1 )
  and this program will append the required data tier.

""")

(options, ARGV) = parser.parse_args()

if len(ARGV) < 1 or len(ARGV) > 2:
    parser.print_help()
    sys.exit(1)


ref_dataset = fixDataSetTier(ARGV.pop(0))

if len(ARGV) > 0:
    new_dataset = fixDataSetTier(ARGV.pop(0))
else:
    new_dataset = None

# example from another validation report:
# https://cmsweb.cern.ch/dqm/offline/start?workspace=HLT;root=HLT/HLTJETMET/ValidationReport;size=L;runnr=0;dataset=/RelValTTbar/CMSSW_3_8_0-MC_38Y_V7-v1/GEN-SIM-RECO;referenceshow=all;referencepos=overlay;referenceobj1=other::/RelValTTbar/CMSSW_3_8_0_pre8-MC_38Y_V6-v1/GEN-SIM-RECO

# there seems not to be a standard URL encoding scheme used here...

for analysisType in knownDatasets.keys():

    dqmdir = knownDatasets[analysisType]['dqmdir']

    parameters = {
        "workspace":      "HLT",
        "root":           "HLT/HLTEgammaValidation/" + dqmdir,
        "size":           "L",
        "runnr":          0,
        "dataset":        ref_dataset,
        "referenceshow":  "all",
        "referencepos":   "overlay",
        }

    if new_dataset != None:
        parameters["referenceobj1"] = "other::" + new_dataset


    print "analysis: " + analysisType
    print
    print "          " + base_url + "?" + ";".join([ x + "=" + urllib.quote(str(parameters[x])) for x in parameters.keys() ])

    print
    print
    
