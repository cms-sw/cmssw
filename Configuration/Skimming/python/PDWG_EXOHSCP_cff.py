TRACK_PT = 20.0
import FWCore.ParameterSet.Config as cms
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi

from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *

generalTracksSkim = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = 'generalTracks',
#	src = 'TrackRefitter',
    filter = False,
    applyBasicCuts = True,
    ptMin = TRACK_PT,
    ptMax = cms.double(999999.0),
    nHitMin = 5,
    chi2nMax = 10.,
)
trackerSeq = cms.Sequence( generalTracksSkim)


from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.TrackProducer.TrackRefitters_cff import *
TrackRefitterSkim = TrackRefitter.clone()
TrackRefitterSkim.src = "generalTracksSkim"


from RecoTracker.DeDx.dedxEstimators_cff import dedxHarmonic2
dedxSkimNPHarm2 = dedxHarmonic2.clone()
dedxSkimNPHarm2.tracks                     = cms.InputTag("TrackRefitterSkim")
dedxSkimNPHarm2.trajectoryTrackAssociation = cms.InputTag("TrackRefitterSkim")
dedxSkimNPHarm2.UsePixel                   = cms.bool(False)

DedxFilter = cms.EDFilter("HSCPFilter",
     inputMuonCollection = cms.InputTag("muons"),
	 inputTrackCollection = cms.InputTag("TrackRefitterSkim"),
	 inputDedxCollection =  cms.InputTag("dedxSkimNPHarm2"),
     SAMuPtMin = cms.double(60),
	 trkPtMin = cms.double(TRACK_PT),
	 dedxMin =cms.double(3.0),
     dedxMaxLeft =cms.double(2.8),
     ndedxHits = cms.int32(3),
     etaMin= cms.double(-2.4),
     etaMax= cms.double(2.4),
     chi2nMax = cms.double(10),
     dxyMax = cms.double(2.0),
     dzMax = cms.double(5),
     filter = cms.bool(True)

)

dedxSeq = cms.Sequence(offlineBeamSpot + MeasurementTrackerEvent + TrackRefitterSkim + dedxSkimNPHarm2+DedxFilter)


from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from TrackingTools.TrackAssociator.default_cfi import *

muonEcalDetIdsEXOHSCP = cms.EDProducer("InterestingEcalDetIdProducer",
								inputCollection = cms.InputTag("muons")
								)
highPtTrackEcalDetIds = cms.EDProducer("HighPtTrackEcalDetIdProducer",
									   #TrackAssociatorParameterBlock
									   TrackAssociatorParameters=TrackAssociatorParameterBlock.TrackAssociatorParameters,
									   inputCollection = cms.InputTag("generalTracksSkim"),
									   TrackPt=cms.double(TRACK_PT)
									   )



detIdProduceSeq = cms.Sequence(muonEcalDetIdsEXOHSCP+highPtTrackEcalDetIds)

reducedHSCPEcalRecHitsEB = cms.EDProducer("ReducedRecHitCollectionProducer",
     recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
     interestingDetIdCollections = cms.VInputTag(
	         #high p_t tracker track ids
	         cms.InputTag("highPtTrackEcalDetIds"),
             #muons
             cms.InputTag("muonEcalDetIdsEXOHSCP")
             ),
     reducedHitsCollection = cms.string('')
)
reducedHSCPEcalRecHitsEE = cms.EDProducer("ReducedRecHitCollectionProducer",
     recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
     interestingDetIdCollections = cms.VInputTag(
	         #high p_t tracker track ids
	         cms.InputTag("highPtTrackEcalDetIds"),
             #muons
             cms.InputTag("muonEcalDetIdsEXOHSCP")
             ),
     reducedHitsCollection = cms.string('')
)


ecalSeq = cms.Sequence(detIdProduceSeq+reducedHSCPEcalRecHitsEB+reducedHSCPEcalRecHitsEE)


reducedHSCPhbhereco = cms.EDProducer("ReduceHcalRecHitCollectionProducer",
									 recHitsLabel = cms.InputTag("hbhereco",""),
									 TrackAssociatorParameters=TrackAssociatorParameterBlock.TrackAssociatorParameters,
									 inputCollection = cms.InputTag("generalTracksSkim"),
									 TrackPt=cms.double(TRACK_PT),
									 reducedHitsCollection = cms.string('')
)

hcalSeq = cms.Sequence(reducedHSCPhbhereco)

muonsSkim = cms.EDProducer("UpdatedMuonInnerTrackRef",
    MuonTag        = cms.untracked.InputTag("muons"),
    OldTrackTag    = cms.untracked.InputTag("generalTracks"),
    NewTrackTag    = cms.untracked.InputTag("generalTracksSkim"),
    maxInvPtDiff   = cms.untracked.double(0.005),
    minDR          = cms.untracked.double(0.01),
)
muonSeq = cms.Sequence(muonsSkim)



TrackAssociatorParametersForHSCPIsol = TrackAssociatorParameterBlock.TrackAssociatorParameters.clone()
TrackAssociatorParametersForHSCPIsol.useHO = cms.bool(False)
TrackAssociatorParametersForHSCPIsol.CSCSegmentCollectionLabel     = cms.InputTag("cscSegments")
TrackAssociatorParametersForHSCPIsol.DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments")
TrackAssociatorParametersForHSCPIsol.EERecHitCollectionLabel       = cms.InputTag("ecalRecHit","EcalRecHitsEE")
TrackAssociatorParametersForHSCPIsol.EBRecHitCollectionLabel       = cms.InputTag("ecalRecHit","EcalRecHitsEB")
TrackAssociatorParametersForHSCPIsol.HBHERecHitCollectionLabel     = cms.InputTag("hbhereco")


HSCPIsolation01 = cms.EDProducer("ProduceIsolationMap",
      inputCollection  = cms.InputTag("generalTracksSkim"),
      IsolationConeDR  = cms.double(0.1),
      TkIsolationPtCut = cms.double(10),
      TKLabel          = cms.InputTag("generalTracks"),
      TrackAssociatorParameters=TrackAssociatorParametersForHSCPIsol,
)

HSCPIsolation03 = HSCPIsolation01.clone()
HSCPIsolation03.IsolationConeDR  = cms.double(0.3)

HSCPIsolation05 = HSCPIsolation01.clone()
HSCPIsolation05.IsolationConeDR  = cms.double(0.5)

exoticaRecoIsoPhotonSeq = cms.EDFilter("MonoPhotonSkimmer",
  phoTag = cms.InputTag("photons::RECO"),
  selectEE = cms.bool(True),
  ecalisoOffsetEB = cms.double(4.2),
  ecalisoSlopeEB = cms.double(0.006),
  hcalisoOffsetEB = cms.double(2.2),
  hcalisoSlopeEB = cms.double(0.0025),
  hadoveremEB = cms.double(0.05),
  minPhoEtEB = cms.double(20.),
  trackIsoOffsetEB = cms.double(2.),
  trackIsoSlopeEB =  cms.double(0.001),
  etaWidthEB  = cms.double(0.013),

  ecalisoOffsetEE = cms.double(4.2),
  ecalisoSlopeEE = cms.double(0.006),
  hcalisoOffsetEE = cms.double(2.2),
  hcalisoSlopeEE = cms.double(0.0025),
  hadoveremEE = cms.double(0.05),
  minPhoEtEE = cms.double(20.),
  trackIsoOffsetEE = cms.double(2.),
  trackIsoSlopeEE =  cms.double(0.001),
  etaWidthEE  = cms.double(0.03),



)


exoticaHSCPSeq = cms.Sequence(trackerSeq+dedxSeq+ecalSeq+hcalSeq+muonSeq+HSCPIsolation01+HSCPIsolation03+HSCPIsolation05)
exoticaHSCPIsoPhotonSeq = cms.Sequence(exoticaRecoIsoPhotonSeq + trackerSeq+ecalSeq+hcalSeq+muonSeq+HSCPIsolation01+HSCPIsolation03+HSCPIsolation05)

EXOHSCPSkim_EventContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "drop *",
      "keep GenEventInfoProduct_generator_*_*",
      "keep L1GlobalTriggerReadoutRecord_*_*_*",
      "keep recoVertexs_offlinePrimaryVertices_*_*",
      "keep recoMuons_muonsSkim_*_*",
      "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
      "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
      "keep recoTracks_generalTracksSkim_*_*",
      "keep recoTrackExtras_generalTracksSkim_*_*",
      "keep TrackingRecHitsOwned_generalTracksSkim_*_*",
      'keep *_dt1DRecHits_*_*',
      'keep *_dt4DSegments_*_*',
      'keep *_csc2DRecHits_*_*',
      'keep *_cscSegments_*_*',
      'keep *_rpcRecHits_*_*',
      'keep recoTracks_standAloneMuons_*_*',
      'keep recoTrackExtras_standAloneMuons_*_*',
      'keep TrackingRecHitsOwned_standAloneMuons_*_*',
      'keep recoTracks_globalMuons_*_*',
      'keep recoTrackExtras_globalMuons_*_*',
      'keep TrackingRecHitsOwned_globalMuons_*_*',
      'keep EcalRecHitsSorted_reducedHSCPEcalRecHitsEB_*_*',
      'keep EcalRecHitsSorted_reducedHSCPEcalRecHitsEE_*_*',
      'keep HBHERecHitsSorted_reducedHSCPhbhereco__*',
      'keep edmTriggerResults_TriggerResults__*',
      'keep *_hltTriggerSummaryAOD_*_*',
      'keep *_HSCPIsolation01__*',
      'keep *_HSCPIsolation03__*',
      'keep *_HSCPIsolation05__*',
      'keep recoPFJets_ak4PFJets__*',
      'keep recoPFMETs_pfMet__*',
      'keep recoBeamSpot_offlineBeamSpot__*',
      )
    )


