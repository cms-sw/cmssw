import FWCore.ParameterSet.Config as cms
from Validation.EventGenerator.BasicGenValidation_cff import *
from FastSimulation.Validation.globalValidation_cff import *
from HLTriggerOffline.Common.HLTValidation_cff import *

from FastSimulation.Configuration.CommonInputs_cff import *
if(MixingMode==2):
#    from SimGeneral.TrackingAnalysis.trackingParticlesFastSim_cfi import *
#    from FastSimulation.Validation.trackingParticlesFastSim_cfi import *
    mergedtruth.mixLabel = cms.string('mixSimCaloHits')
    mergedtruth.simHitLabel = cms.string('g4SimHits')
    mergedtruth.simHitCollections = cms.PSet(tracker = cms.vstring("g4SimHitsTrackerHits"))
    mergedtruthMuon.mixLabel = cms.string('mixSimCaloHits')
    mergedtruthMuon.simHitLabel = cms.string('g4SimHits')
    mergedtruthMuon.simHitCollections = cms.PSet(tracker = cms.vstring("g4SimHitsTrackerHits"))
    TrackAssociatorByHits.ROUList = ['g4SimHitsTrackerHits']
    # reactivate crossing frame, but only if validation is executed:
    from FastSimulation.Configuration.mixHitsWithPU_cfi import *
    mixSimCaloHits.mixObjects.mixSH.crossingFrames = cms.untracked.vstring('MuonCSCHits',
                                                                'MuonDTHits',
                                                                'MuonRPCHits',
                                                                'TrackerHits')
    mixSimCaloHits.mixObjects.mixCH.crossingFrames = cms.untracked.vstring('EcalHitsEB',
                                                       'EcalHitsEE',
                                                       'EcalHitsES',
                                                       'HcalHits')
    mixSimCaloHits.mixObjects.mixTracks.makeCrossingFrame = cms.untracked.bool(True)
    mixSimCaloHits.mixObjects.mixMuonTracks.makeCrossingFrame = cms.untracked.bool(True)
    mixSimCaloHits.mixObjects.mixVertices.makeCrossingFrame = cms.untracked.bool(True)
else:
    mergedtruth.mixLabel = cms.string('mix')
    mergedtruth.simHitLabel = cms.string('famosSimHits')
    mergedtruth.simHitCollections = cms.PSet(tracker = cms.vstring("famosSimHitsTrackerHits"))
    mergedtruthMuon.mixLabel = cms.string('mix')
    mergedtruthMuon.simHitLabel = cms.string('famosSimHits')
    mergedtruthMuon.simHitCollections = cms.PSet(tracker = cms.vstring("famosSimHitsTrackerHits"))
    TrackAssociatorByHits.ROUList = ['famosSimHitsTrackerHits']


prevalidation = cms.Sequence(globalAssociation+hltassociation_fastsim)
prevalidation_preprod = cms.Sequence(globalAssociation)
prevalidation_prod = cms.Sequence(globalAssociation)
validation = cms.Sequence(basicGenTest_seq+globalValidation+hltvalidation_fastsim) 
validation_preprod = cms.Sequence(basicGenTest_seq+globalValidation_preprod+hltvalidation_preprod_fastsim) 
validation_prod = cms.Sequence(basicGenTest_seq+hltvalidation_prod) 
