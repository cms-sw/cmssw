import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Isotrk
# output module 
#  module alcastreamHcalIsotrkNoHLTOutput = PoolOutputModule
OutALCARECOHcalCalIsoTrkFilterNoHLT_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIsoTrkFilterNoHLT')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep HcalNoiseSummary_hcalnoise_*_*',
        'keep *_hbhereco_*_*',
        'keep *_ecalRecHit_*_*',
        'keep *_towerMaker_*_*',
        'keep *_offlineBeamSpot_*_*',
        'keep *_generalTracks_*_*',
        'keep *_generalTracksExtra_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_TkAlIsoProdFilter_*_*',
        'keep *_genParticles_*_*',
        )
)


import copy
OutALCARECOHcalCalIsoTrkFilterNoHLT=copy.deepcopy(OutALCARECOHcalCalIsoTrkFilterNoHLT_noDrop)
OutALCARECOHcalCalIsoTrkFilterNoHLT.outputCommands.insert(0, "drop *")
