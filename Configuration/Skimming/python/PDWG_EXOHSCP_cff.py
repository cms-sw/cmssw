TRACK_PT = 20.0  
import FWCore.ParameterSet.Config as cms
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi

nEventsBefSkim  = cms.EDProducer("EventCountProducer")
beginSeq = cms.Sequence(nEventsBefSkim)

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
HSCPTrigger = hltHighLevel.clone()
HSCPTrigger.andOr = cms.bool( True ) #OR
HSCPTrigger.throw = cms.bool( False )
HSCPTrigger.HLTPaths = [
            "HLT_*_dEdx*",
            "HLT_Mu40_eta2p1*",
            "HLT_Mu50_eta2p1*",
            "HLT_HT650_*",
            "HLT_MET80_*",
            "HLT_PFMET*",
            "HLT_*L2Mu*_NoBPTX*",
            "HLT_L2Mu*_eta2p1_PFMET*",
            "HLT_Mu17_Mu8_v*",
		]
hltSeq = cms.Sequence(HSCPTrigger)  

HSCPdEdxTrigger = HSCPTrigger.clone()
HSCPdEdxTrigger.HLTPaths = [
            "HLT_*_dEdx*"]
hltdEdxSeq = cms.Sequence(HSCPdEdxTrigger) 

#REFIT THE TRACK COLLECTION --> THIS IS NEEDED TO COMPUTE DEDX
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.TrackProducer.TrackRefitters_cff import *
TrackRefitterSkim = TrackRefitter.clone()
TrackRefitterSkim.src = "generalTracks"

#COMPUTE DEDX ON THE TRACK COLLECTION (NO PIXEL NO CLUSTER CLEANING HERE)
dedxSkimNPHarm2 = cms.EDProducer("DeDxEstimatorProducer",
    tracks                     = cms.InputTag("TrackRefitterSkim"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitterSkim"),

    estimator      = cms.string('generic'),
    exponent       = cms.double(-2.0),

    UseStrip       = cms.bool(True),
    UsePixel       = cms.bool(False),
    MeVperADCStrip = cms.double(3.61e-06*265),
    MeVperADCPixel = cms.double(3.61e-06),

    MisCalib_Mean      = cms.untracked.double(1.0),
    MisCalib_Sigma     = cms.untracked.double(0.00),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string(""),
    ShapeTest       = cms.bool(False),
)

#CHECK IF THE EVENT SHOULD BE KEPT OR NOT FOR HSCP
#THREE CASES
#1) Events with HighPT+HighDEdx Tracks are kept
#2) Events with HighPT SA Muon are kept
#3) Events with DoubleMu are kept
HSCPEventFilter = cms.EDFilter("HSCPEventFilter",
     inputMuonCollection = cms.InputTag("muons"),
     inputTrackCollection = cms.InputTag("TrackRefitterSkim"),
     inputDedxCollection =  cms.InputTag("dedxSkimNPHarm2"),

     #USED ONLY FOR HIGH SA MUON SELECTION
     SAMuPtMin = cms.double(70),

     #USED ONLY FOR DIMUON SELECTION
     Mu1PtMin = cms.double(TRACK_PT),
     Mu2PtMin = cms.double(20),

     #USED ONLY FOR HIGH PT AND HIGH DEDX TRACK SELECTION
     trkPtMin = cms.double(TRACK_PT),
     dedxMin =cms.double(3.0),
     dedxMaxLeft =cms.double(2.8),
     ndedxHits = cms.int32(4),
     etaMin= cms.double(-2.4),
     etaMax= cms.double(2.4),
     chi2nMax = cms.double(99999),
     dxyMax = cms.double(99999),
     dzMax = cms.double(99999),
     filter = cms.bool(True)
)

eventSelSeq = cms.Sequence(TrackRefitterSkim + dedxSkimNPHarm2 + HSCPEventFilter)

# AFTER THIS POINT... WE DONT SELECT EVENT ANYMORE... WE ONLY CLEAN OBJECT COLLECTION TO KEEP ONLY THE OBJECTS WE ARE GOING TO USE OFFLINE (REDUCE EVENT SIZE)



#NOW APPLY TIGHT TRACK CLEANING
#FOR INNER TRACKS NOT MATCHING A MUON INNER TRACK: APPLY PT CUT(40GeV) AND DEDX CUT
#FOR INNER TRACKS     MATCHING A MUON INNER TRACK: DON'T APPLY ANY MORE CUTS (than what was already applied by loose selection)
#THE FILTER IS ENABLED WHICH MEANS THAT IF NO TRACKS ARE PASSING THIS LOOSE CLEANING, THE EVENT IS SKIPPED
#OUTPUT OF THIS COLLECTION IS HIGHLY COMPACT AND MUST BE STORE TO THE .ROOT FILE
generalTracksSkim = cms.EDFilter("HSCPTrackSelectorModule",
		 src = cms.InputTag("TrackRefitterSkim"),
		 filter = cms.bool(False),

		 #Keep all muon tracks what ever it's pT
		 muonSource = cms.InputTag("muons"),	

	         #Keep all inner tracks with pt>TRACK_PT whatever it's dEdx
		 trackerTrackPtMin = cms.double(TRACK_PT),
		 usededx = cms.bool(False),
		 InputDedx = cms.InputTag("dedxSkimNPHarm2"),
		 InnerTrackdEdxRightMin = cms.double(-99999.0),
		 InnerTrackdEdxLeftMax = cms.double(99999.0),
		 InnerMuondEdxRightMin = cms.double(-99999.0),
		 InnerMuondEdxLeftMax = cms.double(99999.0),
		 dEdxMeasurementsMinForMuonTrack = cms.uint32(0),
                 dEdxMeasurementsMinForInnerTrack = cms.uint32(4),							 
	)

trackerSeq = cms.Sequence(generalTracksSkim)


from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from TrackingTools.TrackAssociator.default_cfi import *

muonEcalDetIdsHSCP = cms.EDProducer("InterestingEcalDetIdProducer",
		inputCollection = cms.InputTag("muons")
	)
highPtTrackEcalDetIds = cms.EDProducer("HighPtTrackEcalDetIdProducer",
	   #TrackAssociatorParameterBlock
	   TrackAssociatorParameters=TrackAssociatorParameterBlock.TrackAssociatorParameters,
	   inputCollection = cms.InputTag("generalTracksSkim"),
	   TrackPt=cms.double(TRACK_PT)
      )

detIdProduceSeq = cms.Sequence(muonEcalDetIdsHSCP+highPtTrackEcalDetIds)

reducedHSCPEcalRecHitsEB = cms.EDProducer("ReducedRecHitCollectionProducer",
     recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
	 interestingDetIdCollections = cms.VInputTag(
         #high p_t tracker track ids
         cms.InputTag("highPtTrackEcalDetIds"),
         #muons
         cms.InputTag("muonEcalDetIdsHSCP")
     ),
     reducedHitsCollection = cms.string('')
)
reducedHSCPEcalRecHitsEE = cms.EDProducer("ReducedRecHitCollectionProducer",
     recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
     interestingDetIdCollections = cms.VInputTag(
	         #high p_t tracker track ids
	         cms.InputTag("highPtTrackEcalDetIds"),
             #muons
             cms.InputTag("muonEcalDetIdsHSCP")
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

#final sequence
exoticaHSCPSeq = cms.Sequence(beginSeq+hltSeq+eventSelSeq+trackerSeq+ecalSeq+hcalSeq+muonSeq+HSCPIsolation01+HSCPIsolation03+HSCPIsolation05)
exoticaHSCPdEdxSeq = cms.Sequence(beginSeq+hltdEdxSeq+trackerSeq+ecalSeq+hcalSeq+muonSeq+HSCPIsolation01+HSCPIsolation03+HSCPIsolation05)

EXOHSCPSkim_EventContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "drop *",
      "keep EventAux_*_*_*",
      'keep LumiSummary_*_*_*',
      'keep edmMergeableCounter_*_*_*',
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
      'keep *_ak5PFJets_*_*', 
      'keep recoPFMETs_pfMet__*',
       #'keep recoCaloJets_ak5CaloJets__*', #not needed for data
       #'keep recoCaloMETs_met__*',         #not needed for data
      'keep recoBeamSpot_offlineBeamSpot__*',
      )
    )
