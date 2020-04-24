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

dcsonly_json_2012_pickEvents = cms.EDFilter(
    "PickEvents",

    # chose between two definitions for the selection:
    #    run/lumiSection -based with input from a json file (what THIS example does)
    #    run/event -based with input from a json file (the historical PickEvents)

    IsRunLsBased  = cms.bool(True),

    # the file listrunev is unused, in this example
    RunEventList = cms.untracked.string('DPGAnalysis/Skims/data/listrunev'),

    LuminositySectionsBlockRange = LumiList(findFileInPath("DPGAnalysis/Skims/data/json_DCSONLY.txt")).getVLuminosityBlockRange()
    
    )

dcsonly_json_2012 = cms.Sequence( dcsonly_json_2012_pickEvents )
