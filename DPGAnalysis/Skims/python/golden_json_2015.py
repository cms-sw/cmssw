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
    LuminositySectionsBlockRange = LumiList(findFileInPath("DPGAnalysis/Skims/data/Cert_246908-XXXXXX_13TeV_PromptReco_Collisions15_JSON.txt")).getVLuminosityBlockRange()
    
    )

golden_json_2015 = cms.Sequence( golden_json_2015_pickEvents )
