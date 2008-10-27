import FWCore.ParameterSet.Config as cms

# output block for alcastream laserAlignmentT0Producer
# output module 
OutALCARECOTkAlLAS = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlLAS')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_laserAlignmentT0Producer_*_*')
)

