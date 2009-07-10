import FWCore.ParameterSet.Config as cms

# output block for alcastream laserAlignmentT0Producer
# output module 
OutALCARECOTkAlLAS_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlLAS')
    ),
    outputCommands = cms.untracked.vstring(
                                           'keep *_laserAlignmentT0Producer_*_*',
                                           'keep *_MEtoEDMConverter_*_*')
)

import copy
OutALCARECOTkAlLAS = copy.deepcopy(OutALCARECOTkAlLAS_noDrop)
OutALCARECOTkAlLAS.outputCommands.insert(0, "drop *")
