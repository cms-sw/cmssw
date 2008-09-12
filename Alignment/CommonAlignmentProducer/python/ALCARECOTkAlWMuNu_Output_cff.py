import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using WMuNu events
OutALCARECOTkAlWMuNu = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlWMuNu')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
                                           'keep *_ALCARECOTkAlWMuNu_*_*',
                                           'keep *_MEtoEDMConverter_*_*')
)

