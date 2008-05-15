import FWCore.ParameterSet.Config as cms

# tracker
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_RealData_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
#
# calorimeters
#
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import *
from RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi import *
from RecoEcal.Configuration.RecoEcal_cff import *
from Configuration.GlobalRuns.hcal_override_cff import *
#
# muons
from RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff import *
from RecoMuon.Configuration.RecoMuonCosmics_cff import *
#
# jets
#
from RecoJets.Configuration.RecoJets_cff import *
from RecoJets.Configuration.CaloTowersRec_cff import *
from Configuration.GlobalRuns.jet_override_cff import *
trackerGR = cms.Sequence(offlineBeamSpot*siStripZeroSuppression*siStripClusters*siStripMatchedRecHits*tracksP5)
ecal = cms.Sequence(ecalFixedAlphaBetaFitUncalibRecHit*ecalWeightUncalibRecHit*ecalRecHit)
ecalClustersGR = cms.Sequence(islandClusteringSequence*hybridClusteringSequence)
# hcalZeroSuppressedDigis is defined in hcal-override.cff. There is NO cfi in the
# release defining it... amazing!
# also the configuration of hb and ho are in hcal-override but in no cfi stored in CVS
hcal = cms.Sequence(hcalLocalRecoSequence)
caloGR = cms.Sequence(ecal*ecalClustersGR*hcal)
muonsGR = cms.Sequence(muonlocalreco*muonrecoforcosmics)
jetsGR = cms.Sequence(caloTowersRec*recoJets)
reconstructionGR = cms.Sequence(trackerGR*caloGR*muonsGR*jetsGR)
siStripZeroSuppression.RawDigiProducersList = cms.VPSet(cms.PSet(
    RawDigiProducer = cms.string('siStripDigis'),
    RawDigiLabel = cms.string('VirginRaw')
), 
    cms.PSet(
        RawDigiProducer = cms.string('siStripDigis'),
        RawDigiLabel = cms.string('ProcessedRaw')
    ), 
    cms.PSet(
        RawDigiProducer = cms.string('siStripDigis'),
        RawDigiLabel = cms.string('ScopeMode')
    ))
siStripClusters.DigiProducersList = cms.VPSet(cms.PSet(
    DigiLabel = cms.string('ZeroSuppressed'),
    DigiProducer = cms.string('siStripDigis')
), 
    cms.PSet(
        DigiLabel = cms.string('VirginRaw'),
        DigiProducer = cms.string('siStripZeroSuppression')
    ), 
    cms.PSet(
        DigiLabel = cms.string('ProcessedRaw'),
        DigiProducer = cms.string('siStripZeroSuppression')
    ), 
    cms.PSet(
        DigiLabel = cms.string('ScopeMode'),
        DigiProducer = cms.string('siStripZeroSuppression')
    ))

