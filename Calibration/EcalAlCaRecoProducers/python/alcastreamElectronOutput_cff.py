import FWCore.ParameterSet.Config as cms

# output block for alcastream Electron
# output module 
#  module alcastreamElecronOutput = PoolOutputModule
alcastreamElectronOutput = cms.PSet(
    # put this when we have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathAlcastreamElectron')
    ),
    outputCommands = cms.untracked.vstring('drop  *', 
        'keep  TrackCandidates_gsfElectrons_*_*', 
        'keep  *_electronFilter_*_*', 
        'keep  *_alCaIsolatedElectrons_*_*')
)

