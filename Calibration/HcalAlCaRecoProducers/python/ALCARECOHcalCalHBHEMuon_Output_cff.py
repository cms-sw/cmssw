import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL HBHEMuon
# output module 
#  module alcastreamHcalHBHEMuonOutput = PoolOutputModule
OutALCARECOHcalCalHBHEMuon_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalHBHEMuon')
        ),
    outputCommands = cms.untracked.vstring( 
        'keep edmTriggerResults_*_*_*',
        'keep HcalNoiseSummary_hcalnoise_*_*',
        'keep recoTracks_globalMuons_*_*',
        'keep recoTrackExtras_globalMuons_*_*',
        'keep recoTracks_standAloneMuons_*_*',
        'keep recoTrackExtras_standAloneMuons_*_*',
        'keep recoTracks_generalTracks_*_*',
        'keep recoTrackExtras_generalTracks_*_*',
        'keep recoTracks_tevMuons_*_*',
        'keep recoTrackExtras_tevMuons_*_*',
        'keep *_hbhereco_*_*',
        'keep *_hbheprereco_*_*',
        'keep *_HBHEMuonProd_*_*',
        )
)

import copy
OutALCARECOHcalCalHBHEMuon=copy.deepcopy(OutALCARECOHcalCalHBHEMuon_noDrop)
OutALCARECOHcalCalHBHEMuon.outputCommands.insert(0,"drop *")


