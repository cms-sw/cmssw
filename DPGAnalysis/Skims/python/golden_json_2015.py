import FWCore.ParameterSet.Config as cms
from   FWCore.PythonUtilities.LumiList import LumiList
from   os import environ
from   os.path import exists, join

def findFileInPath(theFile):
    for s in environ["CMSSW_SEARCH_PATH"].split(":"):
        attempt = join(s,theFile)
        if exists(attempt):
            return attempt                                                 
    return None

#--------------------------------------------------
#   Pick a set of events
#   defined by a set of run:luminositysection
#--------------------------------------------------

golden_json_2015_pickEvents = cms.EDFilter(
    "PickEvents",

    # chose between two definitions for the selection:
    #    run/lumiSection -based with input from a json file (what THIS example does)
    #    run/event -based with input from a json file (the historical PickEvents)

    IsRunLsBased  = cms.bool(True),

    # the file listrunev is unused, in this example
    RunEventList = cms.untracked.string('DPGAnalysis/Skims/data/listrunev'),

    # the format of the json.txt file is the one of the CMS certification ("Compact list" according to LumiList)
    LuminositySectionsBlockRange = LumiList(findFileInPath("DPGAnalysis/Skims/data/Cert_13TeV_16Dec2015ReReco_Collisions15_25ns_50ns_JSON.txt")).getVLuminosityBlockRange()
    
    )

golden_json_2015 = cms.Sequence( golden_json_2015_pickEvents )
