# /dev/CMSSW_2_1_0_pre6/HLT/V27 (CMSSW_2_1_X_2008-07-03-0200_HLT1)
# Begin replace statements specific to the FastSim HLT
# For all HLTLevel1GTSeed objects, make the following replacements:
#   - L1GtReadoutRecordTag changed from hltGtDigis to gtDigis
#   - L1CollectionsTag changed from l1extraParticles to l1extraParticles
#   - L1MuonCollectionTag changed from l1extraParticles to l1ParamMuons
# For hltL2MuonSeeds: InputObjects and GMTReadoutCollection set to l1ParamMuons
# All other occurances of l1extraParticles recast as l1extraParticles
# L1GtObjectMapTag: hltL1GtObjectMap recast as gtDigis
# L1GtReadoutRecordTag: hltGtDigis recast as gtDigis
# hltMuon[CSC/DT/RPC]Digis changed to muon[CSC/DT/RPC]Digis
# Replace offlineBeamSpot with offlineBeamSpot
# AlCaIsoTrack needs HLTpixelTracking instead of pixelTracks
# Some HLT modules were recast as FastSim sequences: 
#   - hltL3TrackCandidateFromL2, see FastSimulation/HighLevelTrigger/data/Muon/HLTFastRecoForMuon.cff
#   - hltCkfTrackCandidatesL3SingleTau[MET][Relaxed], see FastSimulation/HighLevelTrigger/data/btau/HLTFastRecoForTau.cff
#   - hltCkfTrackCandidatesMumu, see FastSimulation/HighLevelTrigger/data/btau/L3ForDisplacedMumuTrigger.cff
#   - hltCkfTrackCandidatesMumuk, see FastSimulation/HighLevelTrigger/data/btau/L3ForMuMuk.cff
#   - hltBLifetimeRegionalCkfTrackCandidates[Relaxed], see FastSimulation/HighLevelTrigger/data/btau/lifetimeRegionalTracking.cff
# See FastSimulation/Configuration/test/getFastSimHLTcff.py for other documentation
# (L1Menu2007 only) Replace L1_QuadJet30 with L1_QuadJet40
# (Temporary) Remove PSet begin and end from block
# End replace statements specific to the FastSim HLT
# Additional import to make this file self contained
from FastSimulation.HighLevelTrigger.HLTSetup_cff import *

import FWCore.ParameterSet.Config as cms

SiStripQualityFakeESSource = cms.ESSource( "SiStripQualityFakeESSource" )
MCJetCorrectorIcone5 = cms.ESSource( "MCJetCorrectionService",
  tagName = cms.string( "CMSSW_152_iterativeCone5" ),
  label = cms.string( "MCJetCorrectorIcone5" )
)

Chi2EstimatorForL2Refit = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "Chi2EstimatorForL2Refit" ),
  MaxChi2 = cms.double( 1000.0 ),
  nSigma = cms.double( 3.0 )
)
KFTrajectoryFitterForL2Muon = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "KFTrajectoryFitterForL2Muon" ),
  Propagator = cms.string( "SteppingHelixPropagatorAny" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2EstimatorForL2Refit" ),
  minHits = cms.int32( 3 )
)
KFTrajectorySmootherForL2Muon = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "KFTrajectorySmootherForL2Muon" ),
  Propagator = cms.string( "SteppingHelixPropagatorOpposite" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2EstimatorForL2Refit" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 )
)
KFFitterSmootherForL2Muon = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "KFFitterSmootherForL2Muon" ),
  Fitter = cms.string( "KFTrajectoryFitterForL2Muon" ),
  Smoother = cms.string( "KFTrajectorySmootherForL2Muon" ),
  EstimateCut = cms.double( -1.0 ),
  MinNumberOfHits = cms.int32( 5 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False )
)
PixelCPEGenericESProducer = cms.ESProducer( "PixelCPEGenericESProducer",
  ComponentName = cms.string( "PixelCPEGeneric" ),
  eff_charge_cut_lowX = cms.untracked.double( 0.0 ),
  eff_charge_cut_lowY = cms.untracked.double( 0.0 ),
  eff_charge_cut_highX = cms.untracked.double( 1.0 ),
  eff_charge_cut_highY = cms.untracked.double( 1.0 ),
  size_cutX = cms.untracked.double( 3.0 ),
  size_cutY = cms.untracked.double( 3.0 ),
  TanLorentzAnglePerTesla = cms.double( 0.106 ),
  PixelErrorParametrization = cms.string( "NOTcmsim" ),
  Alpha2Order = cms.bool( True )
)
SiStripGainESProducer = cms.ESProducer( "SiStripGainESProducer",
  AutomaticNormalization = cms.bool( False ),
  NormalizationFactor = cms.double( 1.0 ),
  printDebug = cms.untracked.bool( False )
)
SiStripRegionConnectivity = cms.ESProducer( "SiStripRegionConnectivity",
  EtaDivisions = cms.untracked.uint32( 20 ),
  PhiDivisions = cms.untracked.uint32( 20 ),
  EtaMax = cms.untracked.double( 2.5 )
)
bJetRegionalTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "bJetRegionalTrajectoryBuilder" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilterName = cms.string( "bJetRegionalTrajectoryFilter" ),
  maxCand = cms.int32( 1 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False )
)
bJetRegionalTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "bJetRegionalTrajectoryFilter" ),
  filterPset = cms.PSet( 
    minHitsMinPt = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 1.0 ),
    minimumNumberOfHits = cms.int32( 5 ),
    maxNumberOfHits = cms.int32( 8 ),
    maxConsecLostHits = cms.int32( 1 ),
    maxLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" )
  )
)
hltCkfTrajectoryBuilderMumu = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryBuilderMumu" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilterName = cms.string( "hltCkfTrajectoryFilterMumu" ),
  maxCand = cms.int32( 3 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False )
)
hltCkfTrajectoryBuilderMumuk = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryBuilderMumuk" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilterName = cms.string( "hltCkfTrajectoryFilterMumuk" ),
  maxCand = cms.int32( 3 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False )
)
hltCkfTrajectoryFilterMumu = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryFilterMumu" ),
  filterPset = cms.PSet( 
    minHitsMinPt = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 3.0 ),
    minimumNumberOfHits = cms.int32( 5 ),
    maxNumberOfHits = cms.int32( 5 ),
    maxConsecLostHits = cms.int32( 1 ),
    maxLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" )
  )
)
hltCkfTrajectoryFilterMumuk = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryFilterMumuk" ),
  filterPset = cms.PSet( 
    minHitsMinPt = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 3.0 ),
    minimumNumberOfHits = cms.int32( 5 ),
    maxNumberOfHits = cms.int32( 5 ),
    maxConsecLostHits = cms.int32( 1 ),
    maxLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" )
  )
)
pixellayerpairs = cms.ESProducer( "PixelLayerPairsESProducer",
  ComponentName = cms.string( "PixelLayerPairs" ),
  layerList = cms.vstring( 'BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 'BPix1+FPix2_pos', 'BPix1+FPix2_neg', 'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 'BPix2+FPix2_pos', 'BPix2+FPix2_neg', 'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    useErrorsFromParam = cms.untracked.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    useErrorsFromParam = cms.untracked.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    hitErrorRZ = cms.double( 0.0036 )
  )
)
pixellayertriplets = cms.ESProducer( "PixelLayerTripletsESProducer",
  ComponentName = cms.string( "PixelLayerTriplets" ),
  layerList = cms.vstring( 'BPix1+BPix2+BPix3', 'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    useErrorsFromParam = cms.untracked.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    useErrorsFromParam = cms.untracked.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    hitErrorRZ = cms.double( 0.0036 )
  )
)
sistripconn = cms.ESProducer( "SiStripConnectivity" )
softLeptonByDistance = cms.ESProducer( "LeptonTaggerByDistanceESProducer",
  distance = cms.double( 0.5 )
)
softLeptonByPt = cms.ESProducer( "LeptonTaggerByPtESProducer" )
trajBuilderL3 = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "trajBuilderL3" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilterName = cms.string( "trajFilterL3" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False )
)
trajFilterL3 = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "trajFilterL3" ),
  filterPset = cms.PSet( 
    minHitsMinPt = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 0.9 ),
    minimumNumberOfHits = cms.int32( 5 ),
    maxNumberOfHits = cms.int32( 7 ),
    maxConsecLostHits = cms.int32( 1 ),
    maxLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" )
  )
)

hltBoolFirst = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
hltPre1Level1jet15 = cms.EDFilter( "HLTPrescaler" )
hltL1s1Level1jet15 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" ),
    saveTags = cms.untracked.bool( True )
)
hltL1s1jet30 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1jet30 = cms.EDFilter( "HLTPrescaler" )
hltTowerMakerForAll = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.09 ),
    EEThreshold = cms.double( 0.45 ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.9 ),
    HESThreshold = cms.double( 1.4 ),
    HEDThreshold = cms.double( 1.4 ),
    HOThreshold = cms.double( 1.1 ),
    HF1Threshold = cms.double( 1.2 ),
    HF2Threshold = cms.double( 1.8 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( True ),
    MomConstrMethod = cms.int32( 0 ),
    MomEmDepth = cms.double( 0.0 ),
    MomHadDepth = cms.double( 0.0 ),
    MomTotDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    ecalInputs = cms.VInputTag( 'hltEcalRecHitAll:EcalRecHitsEB','hltEcalRecHitAll:EcalRecHitsEE' )
)
hltIterativeCone5CaloJets = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltTowerMakerForAll" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltMCJetCorJetIcone5 = cms.EDProducer( "JetCorrectionProducer",
    src = cms.InputTag( "hltIterativeCone5CaloJets" ),
    alias = cms.untracked.string( "MCJetCorJetIcone5" ),
    correctors = cms.vstring( 'MCJetCorrectorIcone5' )
)
hltMet = cms.EDProducer( "METProducer",
    src = cms.InputTag( "hltTowerMakerForAll" ),
    InputType = cms.string( "CandidateCollection" ),
    METType = cms.string( "CaloMET" ),
    alias = cms.string( "RawCaloMET" ),
    globalThreshold = cms.double( 0.5 ),
    noHF = cms.bool( False )
)
hltHtMet = cms.EDProducer( "METProducer",
    src = cms.InputTag( "hltMCJetCorJetIcone5" ),
    InputType = cms.string( "CaloJetCollection" ),
    METType = cms.string( "MET" ),
    alias = cms.string( "HTMET" ),
    globalThreshold = cms.double( 5.0 ),
    noHF = cms.bool( False )
)
hlt1jet30 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet50 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1jet50 = cms.EDFilter( "HLTPrescaler" )
hlt1jet50 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 50.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet80 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet50" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1jet80 = cms.EDFilter( "HLTPrescaler" )
hltTowerMakerForJets = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.09 ),
    EEThreshold = cms.double( 0.45 ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.9 ),
    HESThreshold = cms.double( 1.4 ),
    HEDThreshold = cms.double( 1.4 ),
    HOThreshold = cms.double( 1.1 ),
    HF1Threshold = cms.double( 1.2 ),
    HF2Threshold = cms.double( 1.8 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( True ),
    MomConstrMethod = cms.int32( 0 ),
    MomEmDepth = cms.double( 0.0 ),
    MomHadDepth = cms.double( 0.0 ),
    MomTotDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    ecalInputs = cms.VInputTag( 'hltEcalRegionalJetsRecHit:EcalRecHitsEB','hltEcalRegionalJetsRecHit:EcalRecHitsEE' )
)
hltIterativeCone5CaloJetsRegional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltTowerMakerForJets" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltMCJetCorJetIcone5Regional = cms.EDProducer( "JetCorrectionProducer",
    src = cms.InputTag( "hltIterativeCone5CaloJetsRegional" ),
    alias = cms.untracked.string( "corJetIcone5" ),
    correctors = cms.vstring( 'MCJetCorrectorIcone5' )
)
hlt1jet80 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 80.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet110 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1jet110 = cms.EDFilter( "HLTPrescaler" )
hlt1jet110 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 110.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet180 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1jet180 = cms.EDFilter( "HLTPrescaler" )
hlt1jet180regional = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 180.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet250 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1jet250 = cms.EDFilter( "HLTPrescaler" )
hlt1jet250 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 250.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1RapGap = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_IsoEG10_Jet15_ForJet10" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrerapgap = cms.EDFilter( "HLTPrescaler" )
hltRapGap = cms.EDFilter( "HLTRapGapFilter",
    inputTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    minEta = cms.double( 3.0 ),
    maxEta = cms.double( 5.0 ),
    caloThresh = cms.double( 20.0 )
)
hltL1s2jet = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150 OR L1_DoubleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre2jet = cms.EDFilter( "HLTPrescaler" )
hlt2jet150 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 150.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltL1s2jetAco = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150 OR L1_DoubleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre2jetAco = cms.EDFilter( "HLTPrescaler" )
hlt2jet125 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 125.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hlt2jetAco = cms.EDFilter( "HLT2JetJet",
    inputTag1 = cms.InputTag( "hlt2jet125" ),
    inputTag2 = cms.InputTag( "hlt2jet125" ),
    saveTags = cms.untracked.bool( True ),
    MinDphi = cms.double( 0.0 ),
    MaxDphi = cms.double( 2.1 ),
    MinDeta = cms.double( 0.0 ),
    MaxDeta = cms.double( -1.0 ),
    MinMinv = cms.double( 0.0 ),
    MaxMinv = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s2jetGap = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre2jetGap = cms.EDFilter( "HLTPrescaler" )
hlt2jetGapFilter = cms.EDFilter( "HLT2jetGapFilter",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    minEt = cms.double( 50.0 ),
    minEta = cms.double( 1.7 )
)
hltL1sdijetave15 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPredijetave15 = cms.EDFilter( "HLTPrescaler" )
hltdijetave15 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    minEtAve = cms.double( 15.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1sdijetave30 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPredijetave30 = cms.EDFilter( "HLTPrescaler" )
hltdijetave30 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    minEtAve = cms.double( 30.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1sdijetave50 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet50" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPredijetave50 = cms.EDFilter( "HLTPrescaler" )
hltdijetave50 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    minEtAve = cms.double( 50.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1sdijetave70 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPredijetave70 = cms.EDFilter( "HLTPrescaler" )
hltdijetave70 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    minEtAve = cms.double( 70.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1sdijetave130 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPredijetave130 = cms.EDFilter( "HLTPrescaler" )
hltdijetave130 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    minEtAve = cms.double( 130.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1sdijetave220 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPredijetave220 = cms.EDFilter( "HLTPrescaler" )
hltdijetave220 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    minEtAve = cms.double( 220.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1s3jet = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre3jet = cms.EDFilter( "HLTPrescaler" )
hlt3jet85 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 85.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltL1s4jet30 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_QuadJet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre4jet30 = cms.EDFilter( "HLTPrescaler" )
hlt4jet30 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltL1s4jet = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre4jet = cms.EDFilter( "HLTPrescaler" )
hlt4jet60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltL1s1SumET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETT60" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1MET1SumET = cms.EDFilter( "HLTPrescaler" )
hlt1SumET120 = cms.EDFilter( "HLTGlobalSumMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTag = cms.untracked.bool( True ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 120.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPreLevel1MET20 = cms.EDFilter( "HLTPrescaler" )
hltL1sLevel1MET20 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM20" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" ),
    saveTags = cms.untracked.bool( True )
)
hltL1s1MET25 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM20" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1MET25 = cms.EDFilter( "HLTPrescaler" )
hlt1MET25 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 25.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1MET35 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM30" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1MET35 = cms.EDFilter( "HLTPrescaler" )
hlt1MET35 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1MET50 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM40" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1MET50 = cms.EDFilter( "HLTPrescaler" )
hlt1MET50 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 50.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1MET65 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM50" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1MET65 = cms.EDFilter( "HLTPrescaler" )
hlt1MET65 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 65.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1MET75 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM50" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1MET75 = cms.EDFilter( "HLTPrescaler" )
hlt1MET75 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 75.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1MET1HT = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT300" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1MET1HT = cms.EDFilter( "HLTPrescaler" )
hlt1HT350 = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    saveTag = cms.untracked.bool( True ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 350.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet1MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1jet1MET = cms.EDFilter( "HLTPrescaler" )
hlt1MET60 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hlt1jet180 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 180.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1sPhiJet1MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrephijet1met = cms.EDFilter( "HLTPrescaler" )
hlt1MET70 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 70.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPhiJet1metAco = cms.EDFilter( "HLTAcoFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    inputMETTag = cms.InputTag( "hlt1MET70" ),
    saveTags = cms.untracked.bool( True ),
    minDeltaPhi = cms.double( 0.0 ),
    maxDeltaPhi = cms.double( 2.89 ),
    minEtJet1 = cms.double( 60.0 ),
    minEtJet2 = cms.double( -1.0 ),
    Acoplanar = cms.string( "Jet1Met" )
)
hltL1s1jet1METAco = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre1jet1METAco = cms.EDFilter( "HLTPrescaler" )
hlt1jet100 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 100.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hlt1jet1METAco = cms.EDFilter( "HLT2JetMET",
    inputTag1 = cms.InputTag( "hlt1jet100" ),
    inputTag2 = cms.InputTag( "hlt1MET60" ),
    saveTags = cms.untracked.bool( True ),
    MinDphi = cms.double( 0.0 ),
    MaxDphi = cms.double( 2.1 ),
    MinDeta = cms.double( 0.0 ),
    MaxDeta = cms.double( -1.0 ),
    MinMinv = cms.double( 0.0 ),
    MaxMinv = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s2jet1MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre2jet1MET = cms.EDFilter( "HLTPrescaler" )
hlt2jet125New = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 125.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltL1s2jetvbfMET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM40" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre2jetvbfMET = cms.EDFilter( "HLTPrescaler" )
hlt2jetvbf = cms.EDFilter( "HLTJetVBFFilter",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    minEt = cms.double( 40.0 ),
    minDeltaEta = cms.double( 2.5 )
)
hltL1sPhi2MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrephi2met = cms.EDFilter( "HLTPrescaler" )
hltPhi2metAco = cms.EDFilter( "HLTPhi2METFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    inputMETTag = cms.InputTag( "hlt1MET60" ),
    saveTags = cms.untracked.bool( True ),
    minDeltaPhi = cms.double( 0.377 ),
    maxDeltaPhi = cms.double( 3.1514 ),
    minEtJet1 = cms.double( 60.0 ),
    minEtJet2 = cms.double( 60.0 )
)
hltL1sPhiJet2MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrephijet2met = cms.EDFilter( "HLTPrescaler" )
hltPhiJet2metAco = cms.EDFilter( "HLTAcoFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    inputMETTag = cms.InputTag( "hlt1MET70" ),
    saveTags = cms.untracked.bool( True ),
    minDeltaPhi = cms.double( 0.377 ),
    maxDeltaPhi = cms.double( 3.141593 ),
    minEtJet1 = cms.double( 50.0 ),
    minEtJet2 = cms.double( 50.0 ),
    Acoplanar = cms.string( "Jet2Met" )
)
hltL1sPhiJet1Jet2 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrephijet1jet2 = cms.EDFilter( "HLTPrescaler" )
hltPhiJet1Jet2Aco = cms.EDFilter( "HLTAcoFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    inputMETTag = cms.InputTag( "hlt1MET70" ),
    saveTags = cms.untracked.bool( True ),
    minDeltaPhi = cms.double( 0.0 ),
    maxDeltaPhi = cms.double( 2.7646 ),
    minEtJet1 = cms.double( 40.0 ),
    minEtJet2 = cms.double( 40.0 ),
    Acoplanar = cms.string( "Jet1Jet2" )
)
hltL1snvMET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrenv = cms.EDFilter( "HLTPrescaler" )
hltnv = cms.EDFilter( "HLTNVFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    inputMETTag = cms.InputTag( "hlt1MET60" ),
    saveTags = cms.untracked.bool( True ),
    minNV = cms.double( 0.1 ),
    minEtJet1 = cms.double( 80.0 ),
    minEtJet2 = cms.double( 20.0 )
)
hltL1s3jet1MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre3jet1MET = cms.EDFilter( "HLTPrescaler" )
hlt3jet60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltL1s4jet1MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPre4jet1MET = cms.EDFilter( "HLTPrescaler" )
hlt4jet35 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltL1seedSingleEgamma = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleIsoEG12" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltIslandBasicClustersEndcapL1Isolated = cms.EDProducer( "EgammaHLTIslandClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    doBarrel = cms.bool( False ),
    doEndcaps = cms.bool( True ),
    doIsolated = cms.bool( True ),
    barrelHitProducer = cms.InputTag( "ecalRecHit" ),
    endcapHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClusters" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClusters" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    l1TagIsolated = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.3 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0_barl = cms.double( 7.4 ),
    posCalc_t0_endc = cms.double( 3.1 ),
    posCalc_t0_endcPresh = cms.double( 1.2 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 )
)
hltIslandBasicClustersBarrelL1Isolated = cms.EDProducer( "EgammaHLTIslandClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( False ),
    doIsolated = cms.bool( True ),
    barrelHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    endcapHitProducer = cms.InputTag( "ecalRecHit" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClusters" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClusters" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    l1TagIsolated = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.3 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0_barl = cms.double( 7.4 ),
    posCalc_t0_endc = cms.double( 3.1 ),
    posCalc_t0_endcPresh = cms.double( 1.2 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 )
)
hltHybridSuperClustersL1Isolated = cms.EDProducer( "EgammaHLTHybridClusterProducer",
    debugLevel = cms.string( "INFO" ),
    basicclusterCollection = cms.string( "" ),
    superclusterCollection = cms.string( "" ),
    ecalhitproducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalhitcollection = cms.string( "EcalRecHitsEB" ),
    l1TagIsolated = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    doIsolated = cms.bool( True ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.14 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0 = cms.double( 7.4 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 ),
    HybridBarrelSeedThr = cms.double( 1.5 ),
    step = cms.int32( 10 ),
    ethresh = cms.double( 0.1 ),
    eseed = cms.double( 0.35 ),
    ewing = cms.double( 1.0 )
)
hltIslandSuperClustersL1Isolated = cms.EDProducer( "SuperClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    endcapClusterProducer = cms.string( "hltIslandBasicClustersEndcapL1Isolated" ),
    barrelClusterProducer = cms.string( "hltIslandBasicClustersBarrelL1Isolated" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClusters" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClusters" ),
    endcapSuperclusterCollection = cms.string( "islandEndcapSuperClusters" ),
    barrelSuperclusterCollection = cms.string( "islandBarrelSuperClusters" ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( True ),
    barrelEtaSearchRoad = cms.double( 0.06 ),
    barrelPhiSearchRoad = cms.double( 0.2 ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    endcapPhiSearchRoad = cms.double( 0.4 ),
    seedTransverseEnergyThreshold = cms.double( 1.5 )
)
hltCorrectedIslandEndcapSuperClustersL1Isolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
    rawSuperClusterProducer = cms.InputTag( 'hltIslandSuperClustersL1Isolated','islandEndcapSuperClusters' ),
    superClusterAlgo = cms.string( "Island" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    etThresh = cms.double( 0.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet( 
      fBremVec = cms.vdouble( 0.0 ),
      fEtEtaVec = cms.vdouble( 0.0 ),
      corrF = cms.vint32( 0 ),
      brLinearLowThr = cms.double( 0.0 ),
      brLinearHighThr = cms.double( 0.0 )
    ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedIslandBarrelSuperClustersL1Isolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( 'hltIslandSuperClustersL1Isolated','islandBarrelSuperClusters' ),
    superClusterAlgo = cms.string( "Island" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    etThresh = cms.double( 0.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet( 
      fBremVec = cms.vdouble( 0.0 ),
      fEtEtaVec = cms.vdouble( 0.0 ),
      corrF = cms.vint32( 0 ),
      brLinearLowThr = cms.double( 0.0 ),
      brLinearHighThr = cms.double( 0.0 )
    ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedHybridSuperClustersL1Isolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( "hltHybridSuperClustersL1Isolated" ),
    superClusterAlgo = cms.string( "Hybrid" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    etThresh = cms.double( 5.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet( 
      fBremVec = cms.vdouble( -0.01217, 0.031, 0.9887, -3.776E-4, 1.598 ),
      fEtEtaVec = cms.vdouble( 1.001, -0.8654, 3.131, 0.0, 0.735, 20.72, 1.169, 8.0, 1.023, -0.00181, 0.0, 1.0, 0.0 ),
      brLinearLowThr = cms.double( 0.7 ),
      brLinearHighThr = cms.double( 8.0 )
    ),
    isl_fCorrPset = cms.PSet(  ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated = cms.EDProducer( "PreshowerClusterProducer",
    preshRecHitProducer = cms.InputTag( 'hltEcalPreshowerRecHit','EcalRecHitsES' ),
    endcapSClusterProducer = cms.InputTag( "hltCorrectedIslandEndcapSuperClustersL1Isolated" ),
    preshClusterCollectionX = cms.string( "preshowerXClusters" ),
    preshClusterCollectionY = cms.string( "preshowerYClusters" ),
    preshNclust = cms.int32( 4 ),
    etThresh = cms.double( 5.0 ),
    preshCalibPlaneX = cms.double( 1.0 ),
    preshCalibPlaneY = cms.double( 0.7 ),
    preshCalibGamma = cms.double( 0.024 ),
    preshCalibMIP = cms.double( 9.0E-5 ),
    assocSClusterCollection = cms.string( "" ),
    preshStripEnergyCut = cms.double( 0.0 ),
    preshSeededNstrip = cms.int32( 15 ),
    preshClusterEnergyCut = cms.double( 0.0 ),
    debugLevel = cms.string( "" )
)
hltL1IsoRecoEcalCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag( "hltCorrectedHybridSuperClustersL1Isolated" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated" ),
    recoEcalCandidateCollection = cms.string( "" )
)
hltL1IsoSingleL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedSingleEgamma" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoSingleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoSingleL1MatchFilter" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsolatedElectronHcalIsol = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.15 )
)
hltL1IsoSingleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSingleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoSingleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoSingleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltPixelMatchElectronsL1Iso = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1IsoWithMaterialTracks" ),
    BSProducer = cms.InputTag( "offlineBeamSpot" )
)
hltL1IsoSingleElectronHOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1IsoSingleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltL1IsoElectronTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    trackProducer = cms.InputTag( "hltL1IsoElectronsRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)
hltL1IsoSingleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoSingleElectronHOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltSingleElectronL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedRelaxedSingleEgamma = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltIslandBasicClustersEndcapL1NonIsolated = cms.EDProducer( "EgammaHLTIslandClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    doBarrel = cms.bool( False ),
    doEndcaps = cms.bool( True ),
    doIsolated = cms.bool( False ),
    barrelHitProducer = cms.InputTag( "ecalRecHit" ),
    endcapHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClusters" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClusters" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    l1TagIsolated = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.3 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0_barl = cms.double( 7.4 ),
    posCalc_t0_endc = cms.double( 3.1 ),
    posCalc_t0_endcPresh = cms.double( 1.2 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 )
)
hltIslandBasicClustersBarrelL1NonIsolated = cms.EDProducer( "EgammaHLTIslandClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( False ),
    doIsolated = cms.bool( False ),
    barrelHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    endcapHitProducer = cms.InputTag( "ecalRecHit" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClusters" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClusters" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    l1TagIsolated = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.3 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0_barl = cms.double( 7.4 ),
    posCalc_t0_endc = cms.double( 3.1 ),
    posCalc_t0_endcPresh = cms.double( 1.2 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 )
)
hltHybridSuperClustersL1NonIsolated = cms.EDProducer( "EgammaHLTHybridClusterProducer",
    debugLevel = cms.string( "INFO" ),
    basicclusterCollection = cms.string( "" ),
    superclusterCollection = cms.string( "" ),
    ecalhitproducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalhitcollection = cms.string( "EcalRecHitsEB" ),
    l1TagIsolated = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    doIsolated = cms.bool( False ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.14 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0 = cms.double( 7.4 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 ),
    HybridBarrelSeedThr = cms.double( 1.5 ),
    step = cms.int32( 10 ),
    ethresh = cms.double( 0.1 ),
    eseed = cms.double( 0.35 ),
    ewing = cms.double( 1.0 )
)
hltIslandSuperClustersL1NonIsolated = cms.EDProducer( "SuperClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    endcapClusterProducer = cms.string( "hltIslandBasicClustersEndcapL1NonIsolated" ),
    barrelClusterProducer = cms.string( "hltIslandBasicClustersBarrelL1NonIsolated" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClusters" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClusters" ),
    endcapSuperclusterCollection = cms.string( "islandEndcapSuperClusters" ),
    barrelSuperclusterCollection = cms.string( "islandBarrelSuperClusters" ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( True ),
    barrelEtaSearchRoad = cms.double( 0.06 ),
    barrelPhiSearchRoad = cms.double( 0.2 ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    endcapPhiSearchRoad = cms.double( 0.4 ),
    seedTransverseEnergyThreshold = cms.double( 1.5 )
)
hltCorrectedIslandEndcapSuperClustersL1NonIsolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
    rawSuperClusterProducer = cms.InputTag( 'hltIslandSuperClustersL1NonIsolated','islandEndcapSuperClusters' ),
    superClusterAlgo = cms.string( "Island" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    etThresh = cms.double( 0.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet( 
      fBremVec = cms.vdouble( 0.0 ),
      fEtEtaVec = cms.vdouble( 0.0 ),
      corrF = cms.vint32( 0 ),
      brLinearLowThr = cms.double( 0.0 ),
      brLinearHighThr = cms.double( 0.0 )
    ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedIslandBarrelSuperClustersL1NonIsolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( 'hltIslandSuperClustersL1NonIsolated','islandBarrelSuperClusters' ),
    superClusterAlgo = cms.string( "Island" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    etThresh = cms.double( 0.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet( 
      fBremVec = cms.vdouble( 0.0 ),
      fEtEtaVec = cms.vdouble( 0.0 ),
      corrF = cms.vint32( 0 ),
      brLinearLowThr = cms.double( 0.0 ),
      brLinearHighThr = cms.double( 0.0 )
    ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedHybridSuperClustersL1NonIsolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( "hltHybridSuperClustersL1NonIsolated" ),
    superClusterAlgo = cms.string( "Hybrid" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    etThresh = cms.double( 5.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet( 
      fBremVec = cms.vdouble( -0.01217, 0.031, 0.9887, -3.776E-4, 1.598 ),
      fEtEtaVec = cms.vdouble( 1.001, -0.8654, 3.131, 0.0, 0.735, 20.72, 1.169, 8.0, 1.023, -0.00181, 0.0, 1.0, 0.0 ),
      brLinearLowThr = cms.double( 0.7 ),
      brLinearHighThr = cms.double( 8.0 )
    ),
    isl_fCorrPset = cms.PSet(  ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated = cms.EDProducer( "PreshowerClusterProducer",
    preshRecHitProducer = cms.InputTag( 'hltEcalPreshowerRecHit','EcalRecHitsES' ),
    endcapSClusterProducer = cms.InputTag( "hltCorrectedIslandEndcapSuperClustersL1NonIsolated" ),
    preshClusterCollectionX = cms.string( "preshowerXClusters" ),
    preshClusterCollectionY = cms.string( "preshowerYClusters" ),
    preshNclust = cms.int32( 4 ),
    etThresh = cms.double( 5.0 ),
    preshCalibPlaneX = cms.double( 1.0 ),
    preshCalibPlaneY = cms.double( 0.7 ),
    preshCalibGamma = cms.double( 0.024 ),
    preshCalibMIP = cms.double( 9.0E-5 ),
    assocSClusterCollection = cms.string( "" ),
    preshStripEnergyCut = cms.double( 0.0 ),
    preshSeededNstrip = cms.int32( 15 ),
    preshClusterEnergyCut = cms.double( 0.0 ),
    debugLevel = cms.string( "" )
)
hltL1NonIsoRecoEcalCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag( "hltCorrectedHybridSuperClustersL1NonIsolated" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated" ),
    recoEcalCandidateCollection = cms.string( "" )
)
hltL1NonIsoSingleElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgamma" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoSingleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoSingleElectronL1MatchFilterRegional" ),
    etcut = cms.double( 18.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsolatedElectronHcalIsol = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.15 )
)
hltL1NonIsoSingleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSingleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoSingleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoSingleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltPixelMatchElectronsL1NonIso = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1NonIsoWithMaterialTracks" ),
    BSProducer = cms.InputTag( "offlineBeamSpot" )
)
hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoSingleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoElectronTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    trackProducer = cms.InputTag( "hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)
hltL1NonIsoSingleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltSingleElectronL1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedRelaxedSingleEgammaEt8 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG8" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
PreHLT1ElectronLW12L1IHIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTIsoSingleElectronEt12L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt8" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTIsoSingleElectronEt12EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTIsoSingleElectronEt12L1MatchFilterRegional" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSingleElectronEt12HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSingleElectronEt12EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSingleElectronEt12PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSingleElectronEt12HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltPixelMatchStartUpElectronsL1Iso = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1IsoStartUpWithMaterialTracks" ),
    BSProducer = cms.InputTag( "offlineBeamSpot" )
)
hltPixelMatchStartUpElectronsL1NonIso = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1NonIsoStartUpWithMaterialTracks" ),
    BSProducer = cms.InputTag( "offlineBeamSpot" )
)
hltL1NonIsoHLTIsoSingleElectronEt12HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSingleElectronEt12PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1IsoStartUpElectronTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    trackProducer = cms.InputTag( "hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)
hltL1NonIsoStartupElectronTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    trackProducer = cms.InputTag( "hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)
hltL1NonIsoHLTIsoSingleElectronEt12TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSingleElectronEt12HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
hltL1IsoLargeWindowSingleL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedSingleEgamma" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoLargeWindowSingleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoLargeWindowSingleL1MatchFilter" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoLargeWindowSingleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoLargeWindowSingleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoLargeWindowSingleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoLargeWindowSingleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1IsoLargeWindowSingleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsL1NonIsoLargeWindowForHLT" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltL1IsoLargeWindowElectronTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    trackProducer = cms.InputTag( "hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)
hltL1IsoLargeWindowSingleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltSingleElectronL1IsoLargeWindowPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgamma" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoLargeWindowSingleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional" ),
    etcut = cms.double( 18.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowSingleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoLargeWindowElectronTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    trackProducer = cms.InputTag( "hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)
hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltSingleElectronL1NonIsoLargeWindowPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedRelaxedSingleEgammaEt12 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG12" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
PreHLT1Electron15L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSingleElectronEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt12" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSingleElectronEt15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronEt15L1MatchFilterRegional" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 6.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronEt15PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronEt15HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronEt15HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronEt15PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSingleElectronEt15TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronEt15HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.12 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
hltL1seedRelaxedSingleEgammaEt15 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
PreHLT1ElectronLWEt18L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSingleElectronLWEt18L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt18EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt18L1MatchFilterRegional" ),
    etcut = cms.double( 18.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt18EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 6.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt18PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt18PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt18TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.12 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
PreHLT1ElectronLWEt15L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSingleElectronLWEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt12" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt15L1MatchFilterRegional" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 6.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt15PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt15PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt15TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.12 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltSingleElectronEt8L1NonIsoHLTnoIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedRelaxedSingleEgammaEt5 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG5" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltL1NonIsoHLTnoIsoSingleElectronEt8L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt5" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTnoIsoSingleElectronEt8EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTnoIsoSingleElectronEt8L1MatchFilterRegional" ),
    etcut = cms.double( 8.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTnoIsoSingleElectronEt8HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTnoIsoSingleElectronEt8EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.0 ),
    hcalisolendcapcut = cms.double( 9999999.0 ),
    HoverEcut = cms.double( 999999.0 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTnoIsoSingleElectronEt8PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTnoIsoSingleElectronEt8HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTnoIsoSingleElectronEt8HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTnoIsoSingleElectronEt8PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTnoIsoSingleElectronEt8TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTnoIsoSingleElectronEt8HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
hltSingleElectronEt10L1NonIsoHLTNonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSingleElectronEt10L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt8" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSingleElectronEt10EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt10L1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonHLTnonIsoIsoSingleElectronEt10HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt10EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.0 ),
    hcalisolendcapcut = cms.double( 9999999.0 ),
    HoverEcut = cms.double( 999999.0 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonHLTnonIsoIsoSingleElectronEt10HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTnonIsoSingleElectronEt10HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSingleElectronEt10TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTnonIsoSingleElectronEt10HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
PreHLT1Electron15L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSingleElectronEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt12" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSingleElectronEt15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt15L1MatchFilterRegional" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.0 ),
    hcalisolendcapcut = cms.double( 9999999.0 ),
    HoverEcut = cms.double( 999999.0 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt15HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt15HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt15HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
PreHLT1Electron18L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSingleElectronEt18L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSingleElectronEt18EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt18L1MatchFilterRegional" ),
    etcut = cms.double( 18.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt18HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt18EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.0 ),
    hcalisolendcapcut = cms.double( 9999999.0 ),
    HoverEcut = cms.double( 999999.0 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt18PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt18HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt18HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt18PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSingleElectronEt18TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt18HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
PreHLT1ElectronLWEt12L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSingleElectronLWEt12L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt8" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt12EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt12L1MatchFilterRegional" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt12HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt12EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt12PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt12HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt12HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt12PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt12TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt12HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.9 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltL1seedRelaxedSingleEgammaEt10 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG10" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
PreHLT1ElectronLWEt15L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSingleElectronLWEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt10" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15L1MatchFilterRegional" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt15HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.9 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltL1NonIsoSingleEMHighEtL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgamma" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoSinglePhotonEMHighEtEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoSingleEMHighEtL1MatchFilterRegional" ),
    etcut = cms.double( 80.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsolatedPhotonEcalIsol = cms.EDProducer( "EgammaHLTEcalIsolationProducersRegional",
    bcBarrelProducer = cms.InputTag( 'hltIslandBasicClustersBarrelL1Isolated','islandBarrelBasicClusters' ),
    bcEndcapProducer = cms.InputTag( 'hltIslandBasicClustersEndcapL1Isolated','islandEndcapBasicClusters' ),
    scIslandBarrelProducer = cms.InputTag( "hltCorrectedIslandBarrelSuperClustersL1Isolated" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated" ),
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    egEcalIsoEtMin = cms.double( 0.0 ),
    egEcalIsoConeSize = cms.double( 0.3 ),
    SCAlgoType = cms.int32( 0 )
)
hltL1NonIsolatedPhotonEcalIsol = cms.EDProducer( "EgammaHLTEcalIsolationProducersRegional",
    bcBarrelProducer = cms.InputTag( 'hltIslandBasicClustersBarrelL1NonIsolated','islandBarrelBasicClusters' ),
    bcEndcapProducer = cms.InputTag( 'hltIslandBasicClustersEndcapL1NonIsolated','islandEndcapBasicClusters' ),
    scIslandBarrelProducer = cms.InputTag( "hltCorrectedIslandBarrelSuperClustersL1NonIsolated" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated" ),
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    egEcalIsoEtMin = cms.double( 0.0 ),
    egEcalIsoConeSize = cms.double( 0.3 ),
    SCAlgoType = cms.int32( 0 )
)
hltL1NonIsoSingleEMHighEtEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEMHighEtEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 5.0 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoSingleEMHighEtHOEFilter = cms.EDFilter( "HLTEgammaHOEFilter",
    candTag = cms.InputTag( "hltL1NonIsoSingleEMHighEtEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 0.05 ),
    hcalisolendcapcut = cms.double( 0.05 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltHcalDoubleCone = cms.EDProducer( "EgammaHLTHcalIsolationDoubleConeProducers",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.3 ),
    egHcalExclusion = cms.double( 0.15 )
)
hltL1NonIsoEMHcalDoubleCone = cms.EDProducer( "EgammaHLTHcalIsolationDoubleConeProducers",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.3 ),
    egHcalExclusion = cms.double( 0.15 )
)
hltL1NonIsoSingleEMHighEtHcalDBCFilter = cms.EDFilter( "HLTEgammaHcalDBCFilter",
    candTag = cms.InputTag( "hltL1NonIsoSingleEMHighEtHOEFilter" ),
    isoTag = cms.InputTag( "hltHcalDoubleCone" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoEMHcalDoubleCone" ),
    hcalisolbarrelcut = cms.double( 8.0 ),
    hcalisolendcapcut = cms.double( 8.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1IsoPhotonTrackIsol = cms.EDProducer( "EgammaHLTPhotonTrackIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    trackProducer = cms.InputTag( "hltL1IsoEgammaRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.3 ),
    egTrkIsoZSpan = cms.double( 999999.0 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.0 )
)
hltL1NonIsoPhotonTrackIsol = cms.EDProducer( "EgammaHLTPhotonTrackIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    trackProducer = cms.InputTag( "hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.3 ),
    egTrkIsoZSpan = cms.double( 999999.0 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.0 )
)
hltL1NonIsoSingleEMHighEtTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSingleEMHighEtHcalDBCFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 4.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltSingleEMVHighEtL1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgamma" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional" ),
    etcut = cms.double( 200.0 ),
    ncandcut = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltSingleEMVHEL1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedDoubleEgamma = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleIsoEG8" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltL1IsoDoubleElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedDoubleEgamma" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoDoubleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoDoubleElectronL1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 9.0 ),
    hcalisolendcapcut = cms.double( 9.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 15000.0 ),
    eoverpendcapcut = cms.double( 24500.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True )
)
hltL1IsoDoubleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.4 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltDoubleElectronL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedRelaxedDoubleEgamma = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG10" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltL1NonIsoDoubleElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEgamma" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoDoubleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoDoubleElectronL1MatchFilterRegional" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoDoubleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoDoubleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9.0 ),
    hcalisolendcapcut = cms.double( 9.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoDoubleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoDoubleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoDoubleElectronEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoDoubleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    eoverpbarrelcut = cms.double( 15000.0 ),
    eoverpendcapcut = cms.double( 24500.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoDoubleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoDoubleElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.4 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltDoubleElectronL1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedDoubleEgamma" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoLargeWindowDoubleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoLargeWindowDoubleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoLargeWindowDoubleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 9.0 ),
    hcalisolendcapcut = cms.double( 9.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoLargeWindowDoubleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoLargeWindowDoubleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoLargeWindowDoubleElectronEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1IsoLargeWindowDoubleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 15000.0 ),
    eoverpendcapcut = cms.double( 24500.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True )
)
hltL1IsoLargeWindowDoubleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoLargeWindowDoubleElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.4 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltDoubleElectronL1IsoLargeWindowPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEgamma" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoLargeWindowDoubleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowDoubleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9.0 ),
    hcalisolendcapcut = cms.double( 9.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoLargeWindowDoubleElectronEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    eoverpbarrelcut = cms.double( 15000.0 ),
    eoverpendcapcut = cms.double( 24500.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowDoubleElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.4 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltDoubleElectronL1NonIsoLargeWindowPresc = cms.EDFilter( "HLTPrescaler" )
hltDoubleElectronEt5L1NonIsoHLTNonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedRelaxedDoubleEgammaEt5 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG5" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEgammaEt5" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional" ),
    etcut = cms.double( 5.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonHLTnonIsoIsoDoubleElectronEt5HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.0 ),
    hcalisolendcapcut = cms.double( 9999999.0 ),
    HoverEcut = cms.double( 999999.0 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonHLTnonIsoIsoDoubleElectronEt5HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTnonIsoDoubleElectronEt5HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTnonIsoDoubleElectronEt5HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
PreHLT2ElectronLWonlyPMEt8L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEgammaEt5" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8L1MatchFilterRegional" ),
    etcut = cms.double( 8.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT2ElectronLWonlyPMEt10L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEgammaEt5" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10L1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1seedRelaxedDoubleEgammaEt10 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG10" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
PreHLT2ElectronLWonlyPMEt12L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEgammaEt10" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12L1MatchFilterRegional" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronZeeL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedDoubleEgamma" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoDoubleElectronZeeEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoDoubleElectronZeeL1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronZeeHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronZeeEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 9.0 ),
    hcalisolendcapcut = cms.double( 9.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronZeePixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronZeeHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronZeeEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronZeePixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 15000.0 ),
    eoverpendcapcut = cms.double( 24500.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True )
)
hltL1IsoDoubleElectronZeeTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronZeeEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.4 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltL1IsoDoubleElectronZeePMMassFilter = cms.EDFilter( "HLTPMMassFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronZeeTrackIsolFilter" ),
    lowerMassCut = cms.double( 54.22 ),
    upperMassCut = cms.double( 99999.9 ),
    nZcandcut = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltZeeCounterPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedExclusiveDoubleEgamma = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ExclusiveDoubleIsoEG6" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltL1IsoDoubleExclElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedExclusiveDoubleEgamma" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoDoubleExclElectronEtPhiFilter = cms.EDFilter( "HLTEgammaDoubleEtPhiFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclElectronL1MatchFilterRegional" ),
    etcut1 = cms.double( 6.0 ),
    etcut2 = cms.double( 6.0 ),
    MinAcop = cms.double( -0.1 ),
    MaxAcop = cms.double( 0.6 ),
    MinEtBalance = cms.double( -1.0 ),
    MaxEtBalance = cms.double( 10.0 ),
    npaircut = cms.int32( 1 )
)
hltL1IsoDoubleExclElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclElectronEtPhiFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 9.0 ),
    hcalisolendcapcut = cms.double( 9.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleExclElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleExclElectronEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1IsoDoubleExclElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 15000.0 ),
    eoverpendcapcut = cms.double( 24500.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True )
)
hltL1IsoDoubleExclElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoDoubleExclElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.4 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltDoubleExclElectronL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedSingleEgammaPrescaled = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleIsoEG10" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltL1IsoSinglePhotonPrescaledL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedSingleEgammaPrescaled" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoSinglePhotonPrescaledEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoSinglePhotonPrescaledL1MatchFilter" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoSinglePhotonPrescaledEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSinglePhotonPrescaledEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonEcalNonIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltL1IsolatedPhotonHcalIsol = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.3 )
)
hltL1IsoSinglePhotonPrescaledHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSinglePhotonPrescaledEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoSinglePhotonPrescaledTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSinglePhotonPrescaledHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonNonIsoTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltSinglePhotonPrescaledL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1IsoSinglePhotonL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedSingleEgamma" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoSinglePhotonEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoSinglePhotonL1MatchFilter" ),
    etcut = cms.double( 30.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoSinglePhotonEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSinglePhotonEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonEcalNonIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltL1IsoSinglePhotonHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSinglePhotonEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoSinglePhotonTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSinglePhotonHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonNonIsoTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltSinglePhotonL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltSinglePhotonEt10L1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoSinglePhotonEt10L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt8" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoSinglePhotonEt10EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt10L1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoSinglePhotonEt10EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt10EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsolatedPhotonHcalIsol = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.3 )
)
hltL1NonIsoSinglePhotonEt10HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt10EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoSinglePhotonEt10TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt10HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon15L1RHIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTIsoSinglePhotonEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt12" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTIsoSinglePhotonEt15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt15L1MatchFilterRegional" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSinglePhotonEt15EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTIsoSinglePhotonEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt15EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSinglePhotonEt15TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt15HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon20L1RHIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTIsoSinglePhotonEt20L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTIsoSinglePhotonEt20EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt20L1MatchFilterRegional" ),
    etcut = cms.double( 20.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSinglePhotonEt20EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt20EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTIsoSinglePhotonEt20HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt20EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSinglePhotonEt20TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt20HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon25L1RHIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTIsoSinglePhotonEt25L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTIsoSinglePhotonEt25EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt25L1MatchFilterRegional" ),
    etcut = cms.double( 25.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSinglePhotonEt25EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt25EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTIsoSinglePhotonEt25HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt25EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSinglePhotonEt25TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt25HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoSinglePhotonL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgamma" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoSinglePhotonEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoSinglePhotonL1MatchFilterRegional" ),
    etcut = cms.double( 40.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoSinglePhotonEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoSinglePhotonHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoSinglePhotonTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltSinglePhotonL1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
PreHLT1Photon20L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSinglePhotonEt20L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt20EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt20L1MatchFilterRegional" ),
    etcut = cms.double( 20.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt20EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt20EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 3.0 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt20HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt20EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 12.0 ),
    hcalisolendcapcut = cms.double( 8.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt20TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt20HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon30L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSinglePhotonEt30L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt30EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt30L1MatchFilterRegional" ),
    etcut = cms.double( 30.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt30EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt30EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 3.0 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt30HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt30EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 12.0 ),
    hcalisolendcapcut = cms.double( 8.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt30TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt30HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon40L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSinglePhotonEt40L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt40EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt40L1MatchFilterRegional" ),
    etcut = cms.double( 40.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt40EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt40EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 3.0 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt40HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt40EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 12.0 ),
    hcalisolendcapcut = cms.double( 8.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt40TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt40HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon45L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSinglePhotonEt45L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt45EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt45L1MatchFilterRegional" ),
    etcut = cms.double( 45.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt45EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt45EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 3.0 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt45HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt45EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 12.0 ),
    hcalisolendcapcut = cms.double( 8.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt45TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt45HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon15L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt10" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon25L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSinglePhotonEt25L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt25EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt25L1MatchFilterRegional" ),
    etcut = cms.double( 25.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt25EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt25EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt25EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt25TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon30L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSinglePhotonEt30L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt30EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt30L1MatchFilterRegional" ),
    etcut = cms.double( 30.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt30EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt30EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt30EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt30TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon40L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSinglePhotonEt40L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEgammaEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt40EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt40L1MatchFilterRegional" ),
    etcut = cms.double( 40.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt40EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt40EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt40HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt40EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt40TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt40HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoublePhotonL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedDoubleEgamma" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoDoublePhotonEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoDoublePhotonL1MatchFilterRegional" ),
    etcut = cms.double( 20.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoublePhotonEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoublePhotonEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonEcalNonIsol" ),
    ecalisolcut = cms.double( 2.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True )
)
hltL1IsoDoublePhotonHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoublePhotonEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 8.0 ),
    hcalisolendcapcut = cms.double( 6.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoublePhotonTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoublePhotonHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonNonIsoTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoublePhotonDoubleEtFilter = cms.EDFilter( "HLTEgammaDoubleEtFilter",
    candTag = cms.InputTag( "hltL1IsoDoublePhotonTrackIsolFilter" ),
    etcut1 = cms.double( 20.0 ),
    etcut2 = cms.double( 20.0 ),
    npaircut = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltDoublePhotonL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoDoublePhotonL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEgamma" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoDoublePhotonEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoDoublePhotonL1MatchFilterRegional" ),
    etcut = cms.double( 20.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoDoublePhotonEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoDoublePhotonEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 2.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoDoublePhotonHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoDoublePhotonEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 8.0 ),
    hcalisolendcapcut = cms.double( 6.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoDoublePhotonTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoDoublePhotonHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoDoublePhotonDoubleEtFilter = cms.EDFilter( "HLTEgammaDoubleEtFilter",
    candTag = cms.InputTag( "hltL1NonIsoDoublePhotonTrackIsolFilter" ),
    etcut1 = cms.double( 20.0 ),
    etcut2 = cms.double( 20.0 ),
    npaircut = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltDoublePhotonL1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
PreHLT2Photon20L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoDoublePhotonEt20L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEgammaEt10" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoDoublePhotonEt20EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoDoublePhotonEt20L1MatchFilterRegional" ),
    etcut = cms.double( 20.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoDoublePhotonEt20EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoDoublePhotonEt20EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 3.0 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoDoublePhotonEt20HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoDoublePhotonEt20EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 12.0 ),
    hcalisolendcapcut = cms.double( 8.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoDoublePhotonEt20TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoDoublePhotonEt20HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT2Photon8L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoublePhotonEt8L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEgammaEt5" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt8EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt8L1MatchFilterRegional" ),
    etcut = cms.double( 8.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt8EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt8EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt8HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt8EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt8TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt8HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT2Photon10L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoublePhotonEt10L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEgammaEt5" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt10EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt10L1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt10EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt10EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt10EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt10TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT2Photon20L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoublePhotonEt20L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEgammaEt10" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt20EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt20L1MatchFilterRegional" ),
    etcut = cms.double( 20.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt20EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt20EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt20HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt20EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt20TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt20HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleExclPhotonL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedExclusiveDoubleEgamma" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoDoubleExclPhotonEtPhiFilter = cms.EDFilter( "HLTEgammaDoubleEtPhiFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclPhotonL1MatchFilterRegional" ),
    etcut1 = cms.double( 10.0 ),
    etcut2 = cms.double( 10.0 ),
    MinAcop = cms.double( -0.1 ),
    MaxAcop = cms.double( 0.6 ),
    MinEtBalance = cms.double( -1.0 ),
    MaxEtBalance = cms.double( 10.0 ),
    npaircut = cms.int32( 1 )
)
hltL1IsoDoubleExclPhotonEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclPhotonEtPhiFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonEcalNonIsol" ),
    ecalisolcut = cms.double( 2.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True )
)
hltL1IsoDoubleExclPhotonHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclPhotonEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 8.0 ),
    hcalisolendcapcut = cms.double( 6.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleExclPhotonTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclPhotonHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonNonIsoTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltDoubleExclPhotonL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltPrescaleMuLevel1Path = cms.EDFilter( "HLTPrescaler" )
hltMuLevel1PathLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7 OR L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltMuLevel1PathL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuLevel1PathLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleMuLevel1Open = cms.EDFilter( "HLTPrescaler" )
hltMuLevel1PathLevel1OpenSeed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuOpen OR L1_SingleMu3 OR L1_SingleMu5" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltMuLevel1PathL1OpenFiltered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuLevel1PathLevel1OpenSeed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuLevel2 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltSingleMuNoIsoL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuNoIsoLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltDt1DRecHits = cms.EDProducer( "DTRecHitProducer",
    debug = cms.untracked.bool( False ),
    dtDigiLabel = cms.InputTag( "simMuonDTDigis" ),
    recAlgo = cms.string( "DTParametrizedDriftAlgo" ),
    recAlgoConfig = cms.PSet( 
      tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
      tTrigModeConfig = cms.PSet( 
        debug = cms.untracked.bool( False ),
        kFactor = cms.double( -2.0 ),
        vPropWire = cms.double( 24.4 ),
        doT0Correction = cms.bool( True ),
        doTOFCorrection = cms.bool( True ),
        tofCorrType = cms.int32( 1 ),
        doWirePropCorrection = cms.bool( True ),
        wirePropCorrType = cms.int32( 1 )
      ),
      minTime = cms.double( -3.0 ),
      maxTime = cms.double( 415.0 ),
      interpolate = cms.bool( True ),
      debug = cms.untracked.bool( False )
    )
)
hltDt4DSegments = cms.EDProducer( "DTRecSegment4DProducer",
    debug = cms.untracked.bool( False ),
    recHits1DLabel = cms.InputTag( "hltDt1DRecHits" ),
    recHits2DLabel = cms.InputTag( "dt2DSegments" ),
    Reco4DAlgoName = cms.string( "DTCombinatorialPatternReco4D" ),
    Reco4DAlgoConfig = cms.PSet( 
      Reco2DAlgoConfig = cms.PSet( 
        recAlgoConfig = cms.PSet( 
          tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
          tTrigModeConfig = cms.PSet( 
            debug = cms.untracked.bool( False ),
            kFactor = cms.double( -2.0 ),
            vPropWire = cms.double( 24.4 ),
            doT0Correction = cms.bool( True ),
            doTOFCorrection = cms.bool( True ),
            tofCorrType = cms.int32( 1 ),
            doWirePropCorrection = cms.bool( True ),
            wirePropCorrType = cms.int32( 1 )
          ),
          minTime = cms.double( -3.0 ),
          maxTime = cms.double( 415.0 ),
          interpolate = cms.bool( True ),
          debug = cms.untracked.bool( False )
        ),
        recAlgo = cms.string( "DTParametrizedDriftAlgo" ),
        MaxAllowedHits = cms.uint32( 50 ),
        AlphaMaxTheta = cms.double( 0.1 ),
        AlphaMaxPhi = cms.double( 1.0 ),
        debug = cms.untracked.bool( False ),
        nSharedHitsMax = cms.int32( 2 ),
        nUnSharedHitsMin = cms.int32( 2 ),
        segmCleanerMode = cms.int32( 1 )
      ),
      Reco2DAlgoName = cms.string( "DTCombinatorialPatternReco" ),
      recAlgoConfig = cms.PSet( 
        tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
        tTrigModeConfig = cms.PSet( 
          debug = cms.untracked.bool( False ),
          kFactor = cms.double( -2.0 ),
          vPropWire = cms.double( 24.4 ),
          doT0Correction = cms.bool( True ),
          doTOFCorrection = cms.bool( True ),
          tofCorrType = cms.int32( 1 ),
          doWirePropCorrection = cms.bool( True ),
          wirePropCorrType = cms.int32( 1 )
        ),
        minTime = cms.double( -3.0 ),
        maxTime = cms.double( 415.0 ),
        interpolate = cms.bool( True ),
        debug = cms.untracked.bool( False )
      ),
      recAlgo = cms.string( "DTParametrizedDriftAlgo" ),
      AllDTRecHits = cms.bool( True ),
      debug = cms.untracked.bool( False ),
      nSharedHitsMax = cms.int32( 2 ),
      nUnSharedHitsMin = cms.int32( 2 ),
      segmCleanerMode = cms.int32( 1 )
    )
)
hltCsc2DRecHits = cms.EDProducer( "CSCRecHitDProducer",
    CSCUseCalibrations = cms.untracked.bool( True ),
    stripDigiTag = cms.InputTag( 'simMuonCSCDigis','MuonCSCStripDigi' ),
    wireDigiTag = cms.InputTag( 'simMuonCSCDigis','MuonCSCWireDigi' ),
    CSCstripWireDeltaTime = cms.untracked.int32( 8 ),
    CSCStripPeakThreshold = cms.untracked.double( 10.0 ),
    CSCStripClusterChargeCut = cms.untracked.double( 25.0 ),
    CSCWireClusterDeltaT = cms.untracked.int32( 1 ),
    CSCStripxtalksOffset = cms.untracked.double( 0.03 ),
    NoiseLevel_ME1a = cms.untracked.double( 7.0 ),
    XTasymmetry_ME1a = cms.untracked.double( 0.0 ),
    ConstSyst_ME1a = cms.untracked.double( 0.022 ),
    NoiseLevel_ME1b = cms.untracked.double( 7.0 ),
    XTasymmetry_ME1b = cms.untracked.double( 0.0 ),
    ConstSyst_ME1b = cms.untracked.double( 0.02 ),
    NoiseLevel_ME12 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME12 = cms.untracked.double( 0.025 ),
    ConstSyst_ME12 = cms.untracked.double( 0.045 ),
    NoiseLevel_ME13 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME13 = cms.untracked.double( 0.025 ),
    ConstSyst_ME13 = cms.untracked.double( 0.065 ),
    NoiseLevel_ME21 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME21 = cms.untracked.double( 0.025 ),
    ConstSyst_ME21 = cms.untracked.double( 0.06 ),
    NoiseLevel_ME22 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME22 = cms.untracked.double( 0.025 ),
    ConstSyst_ME22 = cms.untracked.double( 0.06 ),
    NoiseLevel_ME31 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME31 = cms.untracked.double( 0.025 ),
    ConstSyst_ME31 = cms.untracked.double( 0.06 ),
    NoiseLevel_ME32 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME32 = cms.untracked.double( 0.025 ),
    ConstSyst_ME32 = cms.untracked.double( 0.06 ),
    NoiseLevel_ME41 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME41 = cms.untracked.double( 0.025 ),
    ConstSyst_ME41 = cms.untracked.double( 0.06 ),
    readBadChannels = cms.bool( False )
)
hltCscSegments = cms.EDProducer( "CSCSegmentProducer",
    inputObjects = cms.InputTag( "hltCsc2DRecHits" ),
    algo_type = cms.int32( 4 ),
    algo_psets = cms.VPSet( 
      cms.PSet(  algo_name = cms.string( "CSCSegAlgoSK" ),
        parameters_per_chamber_type = cms.vint32( 2, 1, 1, 1, 1, 1, 1, 1, 1 ),
        chamber_types = cms.vstring( 'ME1/a', 'ME1/b', 'ME1/2', 'ME1/3', 'ME2/1', 'ME2/2', 'ME3/1', 'ME3/2', 'ME4/1' ),
        algo_psets = cms.VPSet( 
          cms.PSet(  minLayersApart = cms.int32( 2 ),
            wideSeg = cms.double( 3.0 ),
            chi2Max = cms.double( 99999.0 ),
            dPhiFineMax = cms.double( 0.025 ),
            dRPhiFineMax = cms.double( 8.0 ),
            dPhiMax = cms.double( 0.0030 ),
            dRPhiMax = cms.double( 8.0 ),
            verboseInfo = cms.untracked.bool( True )
          ),
          cms.PSet(  minLayersApart = cms.int32( 2 ),
            wideSeg = cms.double( 3.0 ),
            chi2Max = cms.double( 99999.0 ),
            dPhiFineMax = cms.double( 0.025 ),
            dRPhiFineMax = cms.double( 3.0 ),
            dPhiMax = cms.double( 0.025 ),
            dRPhiMax = cms.double( 8.0 ),
            verboseInfo = cms.untracked.bool( True )
          )
        )
      ),
      cms.PSet(  algo_name = cms.string( "CSCSegAlgoTC" ),
        parameters_per_chamber_type = cms.vint32( 2, 1, 1, 1, 1, 1, 1, 1, 1 ),
        chamber_types = cms.vstring( 'ME1/a', 'ME1/b', 'ME1/2', 'ME1/3', 'ME2/1', 'ME2/2', 'ME3/1', 'ME3/2', 'ME4/1' ),
        algo_psets = cms.VPSet( 
          cms.PSet(  SegmentSorting = cms.int32( 1 ),
            minLayersApart = cms.int32( 2 ),
            chi2ndfProbMin = cms.double( 1.0E-4 ),
            chi2Max = cms.double( 6000.0 ),
            dPhiFineMax = cms.double( 0.02 ),
            dRPhiFineMax = cms.double( 6.0 ),
            dPhiMax = cms.double( 0.0030 ),
            dRPhiMax = cms.double( 1.2 ),
            verboseInfo = cms.untracked.bool( True )
          ),
          cms.PSet(  SegmentSorting = cms.int32( 1 ),
            minLayersApart = cms.int32( 2 ),
            chi2ndfProbMin = cms.double( 1.0E-4 ),
            chi2Max = cms.double( 6000.0 ),
            dPhiFineMax = cms.double( 0.013 ),
            dRPhiFineMax = cms.double( 3.0 ),
            dPhiMax = cms.double( 0.00198 ),
            dRPhiMax = cms.double( 0.6 ),
            verboseInfo = cms.untracked.bool( True )
          )
        )
      ),
      cms.PSet(  algo_name = cms.string( "CSCSegAlgoDF" ),
        parameters_per_chamber_type = cms.vint32( 3, 1, 2, 2, 1, 2, 1, 2, 1 ),
        chamber_types = cms.vstring( 'ME1/a', 'ME1/b', 'ME1/2', 'ME1/3', 'ME2/1', 'ME2/2', 'ME3/1', 'ME3/2', 'ME4/1' ),
        algo_psets = cms.VPSet( 
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            Pruning = cms.untracked.bool( False ),
            nHitsPerClusterIsShower = cms.int32( 20 ),
            minHitsForPreClustering = cms.int32( 10 ),
            dYclusBoxMax = cms.double( 8.0 ),
            dXclusBoxMax = cms.double( 8.0 ),
            preClustering = cms.untracked.bool( False ),
            chi2Max = cms.double( 5000.0 ),
            tanPhiMax = cms.double( 0.5 ),
            tanThetaMax = cms.double( 1.2 ),
            minLayersApart = cms.int32( 2 ),
            dPhiFineMax = cms.double( 0.025 ),
            dRPhiFineMax = cms.double( 8.0 ),
            minHitsPerSegment = cms.int32( 3 ),
            CSCSegmentDebug = cms.untracked.bool( False )
          ),
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            Pruning = cms.untracked.bool( False ),
            nHitsPerClusterIsShower = cms.int32( 20 ),
            minHitsForPreClustering = cms.int32( 10 ),
            dYclusBoxMax = cms.double( 12.0 ),
            dXclusBoxMax = cms.double( 8.0 ),
            preClustering = cms.untracked.bool( False ),
            chi2Max = cms.double( 5000.0 ),
            tanPhiMax = cms.double( 0.8 ),
            tanThetaMax = cms.double( 2.0 ),
            minLayersApart = cms.int32( 2 ),
            dPhiFineMax = cms.double( 0.025 ),
            dRPhiFineMax = cms.double( 12.0 ),
            minHitsPerSegment = cms.int32( 3 ),
            CSCSegmentDebug = cms.untracked.bool( False )
          ),
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            Pruning = cms.untracked.bool( False ),
            nHitsPerClusterIsShower = cms.int32( 20 ),
            minHitsForPreClustering = cms.int32( 30 ),
            dYclusBoxMax = cms.double( 8.0 ),
            dXclusBoxMax = cms.double( 8.0 ),
            preClustering = cms.untracked.bool( False ),
            chi2Max = cms.double( 5000.0 ),
            tanPhiMax = cms.double( 0.5 ),
            tanThetaMax = cms.double( 1.2 ),
            minLayersApart = cms.int32( 2 ),
            dPhiFineMax = cms.double( 0.025 ),
            dRPhiFineMax = cms.double( 8.0 ),
            minHitsPerSegment = cms.int32( 3 ),
            CSCSegmentDebug = cms.untracked.bool( False )
          )
        )
      ),
      cms.PSet(  algo_name = cms.string( "CSCSegAlgoST" ),
        parameters_per_chamber_type = cms.vint32( 2, 1, 1, 1, 1, 1, 1, 1, 1 ),
        chamber_types = cms.vstring( 'ME1/a', 'ME1/b', 'ME1/2', 'ME1/3', 'ME2/1', 'ME2/2', 'ME3/1', 'ME3/2', 'ME4/1' ),
        algo_psets = cms.VPSet( 
          cms.PSet(  curvePenalty = cms.untracked.double( 2.0 ),
            curvePenaltyThreshold = cms.untracked.double( 0.85 ),
            yweightPenalty = cms.untracked.double( 1.5 ),
            yweightPenaltyThreshold = cms.untracked.double( 1.0 ),
            hitDropLimit6Hits = cms.untracked.double( 0.3333 ),
            hitDropLimit5Hits = cms.untracked.double( 0.8 ),
            hitDropLimit4Hits = cms.untracked.double( 0.6 ),
            onlyBestSegment = cms.untracked.bool( False ),
            BrutePruning = cms.untracked.bool( False ),
            Pruning = cms.untracked.bool( False ),
            preClustering = cms.untracked.bool( True ),
            maxRecHitsInCluster = cms.untracked.int32( 20 ),
            dYclusBoxMax = cms.untracked.double( 8.0 ),
            dXclusBoxMax = cms.untracked.double( 4.0 ),
            minHitsPerSegment = cms.untracked.int32( 3 ),
            CSCDebug = cms.untracked.bool( False )
          ),
          cms.PSet(  curvePenalty = cms.untracked.double( 2.0 ),
            curvePenaltyThreshold = cms.untracked.double( 0.85 ),
            yweightPenalty = cms.untracked.double( 1.5 ),
            yweightPenaltyThreshold = cms.untracked.double( 1.0 ),
            hitDropLimit6Hits = cms.untracked.double( 0.3333 ),
            hitDropLimit5Hits = cms.untracked.double( 0.8 ),
            hitDropLimit4Hits = cms.untracked.double( 0.6 ),
            onlyBestSegment = cms.untracked.bool( False ),
            BrutePruning = cms.untracked.bool( False ),
            Pruning = cms.untracked.bool( False ),
            preClustering = cms.untracked.bool( True ),
            maxRecHitsInCluster = cms.untracked.int32( 24 ),
            dYclusBoxMax = cms.untracked.double( 8.0 ),
            dXclusBoxMax = cms.untracked.double( 4.0 ),
            minHitsPerSegment = cms.untracked.int32( 3 ),
            CSCDebug = cms.untracked.bool( False )
          )
        )
      )
    )
)
hltRpcRecHits = cms.EDProducer( "RPCRecHitProducer",
    rpcDigiLabel = cms.InputTag( "simMuonRPCDigis" ),
    recAlgo = cms.string( "RPCRecHitStandardAlgo" ),
    recAlgoConfig = cms.PSet(  )
)
hltL2MuonSeeds = cms.EDProducer( "L2MuonSeedGenerator",
    InputObjects = cms.InputTag( "l1ParamMuons" ),
    GMTReadoutCollection = cms.InputTag( "l1ParamMuons" ),
    Propagator = cms.string( "SteppingHelixPropagatorAny" ),
    L1MinPt = cms.double( 0.0 ),
    L1MaxEta = cms.double( 2.5 ),
    L1MinQuality = cms.uint32( 1 ),
    ServiceParameters = cms.PSet( 
      UseMuonNavigation = cms.untracked.bool( True ),
      RPCLayers = cms.bool( True ),
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny', 'SteppingHelixPropagatorAlong', 'SteppingHelixPropagatorOpposite', 'PropagatorWithMaterial', 'PropagatorWithMaterialOpposite', 'SmartPropagator', 'SmartPropagatorOpposite', 'SmartPropagatorAnyOpposite', 'SmartPropagatorAny', 'SmartPropagatorRK', 'SmartPropagatorAnyRK' )
    )
)
hltL2Muons = cms.EDProducer( "L2MuonProducer",
    InputObjects = cms.InputTag( "hltL2MuonSeeds" ),
    ServiceParameters = cms.PSet( 
      UseMuonNavigation = cms.untracked.bool( True ),
      RPCLayers = cms.bool( True ),
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny', 'SteppingHelixPropagatorAlong', 'SteppingHelixPropagatorOpposite', 'PropagatorWithMaterial', 'PropagatorWithMaterialOpposite', 'SmartPropagator', 'SmartPropagatorOpposite', 'SmartPropagatorAnyOpposite', 'SmartPropagatorAny', 'SmartPropagatorRK', 'SmartPropagatorAnyRK' )
    ),
    TrackLoaderParameters = cms.PSet( 
      MuonUpdatorAtVertexParameters = cms.PSet( 
        Propagator = cms.string( "SteppingHelixPropagatorOpposite" ),
        BeamSpotPosition = cms.vdouble( 0.0, 0.0, 0.0 ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 ),
        MaxChi2 = cms.double( 1000000.0 )
      ),
      VertexConstraint = cms.bool( True ),
      DoSmoothing = cms.bool( False ),
      Smoother = cms.string( "KFSmootherForMuonTrackLoader" )
    ),
    L2TrajBuilderParameters = cms.PSet( 
      SeedPosition = cms.string( "in" ),
      SeedPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      NavigationType = cms.string( "Standard" ),
      RefitterParameters = cms.PSet( 
        FitterName = cms.string( "KFFitterSmootherForL2Muon" ),
        Option = cms.int32( 1 )
      ),
      BWFilterParameters = cms.PSet( 
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        EnableDTMeasurement = cms.bool( True ),
        EnableCSCMeasurement = cms.bool( True ),
        EnableRPCMeasurement = cms.bool( True ),
        BWSeedType = cms.string( "fromGenerator" ),
        FitDirection = cms.string( "outsideIn" ),
        Propagator = cms.string( "SteppingHelixPropagatorAny" ),
        MaxChi2 = cms.double( 25.0 ),
        NumberOfSigma = cms.double( 3.0 ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          MaxChi2 = cms.double( 25.0 ),
          Granularity = cms.int32( 2 ),
          RescaleError = cms.bool( False ),
          RescaleErrorFactor = cms.double( 100.0 )
        )
      ),
      DoRefit = cms.bool( False ),
      FilterParameters = cms.PSet( 
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        EnableDTMeasurement = cms.bool( True ),
        EnableCSCMeasurement = cms.bool( True ),
        EnableRPCMeasurement = cms.bool( True ),
        FitDirection = cms.string( "insideOut" ),
        Propagator = cms.string( "SteppingHelixPropagatorAny" ),
        MaxChi2 = cms.double( 1000.0 ),
        NumberOfSigma = cms.double( 3.0 ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          MaxChi2 = cms.double( 1000.0 ),
          Granularity = cms.int32( 0 ),
          RescaleError = cms.bool( False ),
          RescaleErrorFactor = cms.double( 100.0 )
        )
      ),
      DoBackwardFilter = cms.bool( True )
    )
)
hltL2MuonCandidates = cms.EDProducer( "L2MuonCandidateProducer",
    InputObjects = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
hltSingleMuLevel2NoIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 9.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuIso9 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuIsoLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltSingleMuIsoL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuIsoLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuIsoL2PreFiltered7 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltTowerMakerForMuons = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.09 ),
    EEThreshold = cms.double( 0.45 ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.9 ),
    HESThreshold = cms.double( 1.4 ),
    HEDThreshold = cms.double( 1.4 ),
    HOThreshold = cms.double( 1.1 ),
    HF1Threshold = cms.double( 1.2 ),
    HF2Threshold = cms.double( 1.8 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( True ),
    MomConstrMethod = cms.int32( 0 ),
    MomEmDepth = cms.double( 0.0 ),
    MomHadDepth = cms.double( 0.0 ),
    MomTotDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    ecalInputs = cms.VInputTag( 'hltEcalRegionalMuonsRecHit:EcalRecHitsEB','hltEcalRegionalMuonsRecHit:EcalRecHitsEE' )
)
hltL2MuonIsolations = cms.EDProducer( "L2MuonIsolationProducer",
    StandAloneCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    OutputMuIsoDeposits = cms.bool( True ),
    EtaBounds = cms.vdouble( 0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 1.785, 1.88, 1.9865, 2.1075, 2.247, 2.411 ),
    ConeSizes = cms.vdouble( 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24 ),
    Thresholds = cms.vdouble( 4.0, 3.7, 4.0, 3.5, 3.4, 3.4, 3.2, 3.4, 3.1, 2.9, 2.9, 2.7, 3.1, 3.0, 2.4, 2.1, 2.0, 2.3, 2.2, 2.4, 2.5, 2.5, 2.6, 2.9, 3.1, 2.9 ),
    ExtractorPSet = cms.PSet( 
      ComponentName = cms.string( "CaloExtractor" ),
      CaloTowerCollectionLabel = cms.InputTag( "hltTowerMakerForMuons" ),
      DepositLabel = cms.untracked.string( "EcalPlusHcal" ),
      Weight_E = cms.double( 1.5 ),
      Weight_H = cms.double( 1.0 ),
      Threshold_E = cms.double( 0.2 ),
      Threshold_H = cms.double( 0.5 ),
      DR_Veto_E = cms.double( 0.07 ),
      DR_Veto_H = cms.double( 0.1 ),
      DR_Max = cms.double( 0.24 ),
      Vertex_Constraint_XY = cms.bool( False ),
      Vertex_Constraint_Z = cms.bool( False )
    )
)
hltSingleMuIsoL2IsoFiltered7 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL2PreFiltered7" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
    InputObjects = cms.InputTag( "hltL3Muons" )
)
hltSingleMuIsoL3PreFiltered9 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL2IsoFiltered7" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 9.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltL3MuonIsolations = cms.EDProducer( "L3MuonIsolationProducer",
    inputMuonCollection = cms.InputTag( "hltL3Muons" ),
    OutputMuIsoDeposits = cms.bool( True ),
    TrackPt_Min = cms.double( -1.0 ),
    ExtractorPSet = cms.PSet( 
      ComponentName = cms.string( "TrackExtractor" ),
      inputTrackCollection = cms.InputTag( "hltPixelTracks" ),
      DepositLabel = cms.untracked.string( "PXLS" ),
      Diff_r = cms.double( 0.1 ),
      Diff_z = cms.double( 0.2 ),
      DR_Veto = cms.double( 0.01 ),
      DR_Max = cms.double( 0.24 ),
      BeamlineOption = cms.string( "BeamSpotFromEvent" ),
      BeamSpotLabel = cms.InputTag( "offlineBeamSpot" ),
      NHits_Min = cms.uint32( 0 ),
      Chi2Ndof_Max = cms.double( 1.0E64 ),
      Chi2Prob_Min = cms.double( -1.0 ),
      Pt_Min = cms.double( -1.0 )
    ),
    CutsPSet = cms.PSet( 
      ComponentName = cms.string( "SimpleCuts" ),
      EtaBounds = cms.vdouble( 0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 1.785, 1.88, 1.9865, 2.1075, 2.247, 2.411 ),
      ConeSizes = cms.vdouble( 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24 ),
      Thresholds = cms.vdouble( 1.1, 1.1, 1.1, 1.1, 1.2, 1.1, 1.2, 1.1, 1.2, 1.0, 1.1, 1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 0.9, 1.1, 0.9, 1.1, 1.0, 1.0, 0.9, 0.8, 0.1 ),
      applyCutsORmaxNTracks = cms.bool( False ),
      maxNTracks = cms.int32( -1 )
    )
)
hltSingleMuIsoL3IsoFiltered9 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL3PreFiltered9" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuIso = cms.EDFilter( "HLTPrescaler" )
hltSingleMuIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 9.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuIsoL2IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL2PreFiltered" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltSingleMuIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL2IsoFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 11.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuIsoL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltPrescaleSingleMuIso13 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuIsoLevel1Seed10 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu10" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltSingleMuIsoL1Filtered10 = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuIsoLevel1Seed10" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuIsoL2PreFiltered11 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL1Filtered10" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 11.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuIsoL2IsoFiltered11 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL2PreFiltered11" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltSingleMuIsoL3PreFiltered13 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL2IsoFiltered11" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 13.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuIsoL3IsoFiltered13 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL3PreFiltered13" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuIso15 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuIsoL2PreFiltered12 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL1Filtered10" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 12.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuIsoL2IsoFiltered12 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL2PreFiltered12" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltSingleMuIsoL3PreFiltered15 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL2IsoFiltered12" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuIsoL3IsoFiltered15 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL3PreFiltered15" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIso = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 12.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltL3TkMuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
    InputObjects = cms.InputTag( "hltL3TkMuons" )
)
hltSingleMuNoIsoL3TkPreFilter = cms.EDFilter( "HLTMuonL3TkPreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3TkMuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 99999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltPrescaleSingleMuPrescale3 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuPrescale3Level1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltSingleMuPrescale3L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuPrescale3Level1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuPrescale3L2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale3L1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuPrescale3L3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale3L2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuPrescale5 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuPrescale5Level1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu5" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltSingleMuPrescale5L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuPrescale5Level1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuPrescale5L2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale5L1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuPrescale5L3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale5L2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPreSingleMuPrescale77 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuPrescale77Level1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltSingleMuPrescale77L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuPrescale77Level1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuPrescale77L2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale77L1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuPrescale77L3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale77L2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIso9 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoL2PreFiltered7 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuNoIsoL3PreFiltered9 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered7" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 9.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPreSingleMuPrescale710 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuPrescale710Level1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltSingleMuPrescale710L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuPrescale710Level1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuPrescale710L2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale710L1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 8.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuPrescale710L3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale710L2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 10.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIso11 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoL2PreFiltered9 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 9.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuNoIsoL3PreFiltered11 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered9" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 11.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIso13 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoLevel1Seed10 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu10" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltSingleMuNoIsoL1Filtered10 = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuNoIsoLevel1Seed10" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuNoIsoL2PreFiltered11 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL1Filtered10" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 11.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuNoIsoL3PreFiltered13 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered11" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 13.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIso15 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoL2PreFiltered12 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL1Filtered10" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 12.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuNoIsoL3PreFiltered15 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered12" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltSingleMuNoIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIsoRelaxedVtx2cm = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIsoRelaxedVtx2mm = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoL3PreFilteredRelaxedVtx2mm = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 0.2 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleDiMuonIso = cms.EDFilter( "HLTPrescaler" )
hltDiMuonIsoLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltDiMuonIsoL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltDiMuonIsoLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltDiMuonIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonIsoL1Filtered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltDiMuonIsoL2IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltDiMuonIsoL2PreFiltered" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltDiMuonIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonIsoL2IsoFiltered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltDiMuonIsoL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltDiMuonIsoL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleDiMuonNoIso = cms.EDFilter( "HLTPrescaler" )
hltDiMuonNoIsoLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltDiMuonNoIsoL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltDiMuonNoIsoLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltDiMuonNoIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonNoIsoL1Filtered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltDiMuonNoIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonNoIsoL2PreFiltered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleDiMuonNoIsoRelaxedVtx2cm = cms.EDFilter( "HLTPrescaler" )
hltDiMuonNoIsoL3PreFilteredRelaxedVtx2cm = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonNoIsoL2PreFiltered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleDiMuonNoIsoRelaxedVtx2mm = cms.EDFilter( "HLTPrescaler" )
hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonNoIsoL2PreFiltered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 0.2 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleJPsiMM = cms.EDFilter( "HLTPrescaler" )
hltJpsiMMLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltJpsiMML1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltJpsiMMLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltJpsiMML2Filtered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltJpsiMML1Filtered" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 100.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 1.0 ),
    MaxInvMass = cms.double( 5.0 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltJpsiMML3Filtered = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltJpsiMML2Filtered" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 2.8 ),
    MaxInvMass = cms.double( 3.4 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleUpsilonMM = cms.EDFilter( "HLTPrescaler" )
hltUpsilonMMLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltUpsilonMML1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltUpsilonMMLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltUpsilonMML2Filtered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltUpsilonMML1Filtered" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 100.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 6.0 ),
    MaxInvMass = cms.double( 13.0 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltUpsilonMML3Filtered = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltUpsilonMML2Filtered" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 8.0 ),
    MaxInvMass = cms.double( 11.0 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleZMM = cms.EDFilter( "HLTPrescaler" )
hltZMMLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltZMML1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltZMMLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltZMML2Filtered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltZMML1Filtered" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 100.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 7.0 ),
    MinPtMin = cms.double( 7.0 ),
    MinInvMass = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltZMML3Filtered = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltZMML2Filtered" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 7.0 ),
    MinPtMin = cms.double( 7.0 ),
    MinInvMass = cms.double( 70.0 ),
    MaxInvMass = cms.double( 1.0E30 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSameSignMu = cms.EDFilter( "HLTPrescaler" )
hltSameSignMuLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltSameSignMuL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSameSignMuLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltSameSignMuL2PreFiltered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltSameSignMuL1Filtered" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 100.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 1 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSameSignMuL3PreFiltered = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSameSignMuL2PreFiltered" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 1 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescalePsi2SMM = cms.EDFilter( "HLTPrescaler" )
hltPsi2SMML2Filtered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltJpsiMML1Filtered" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 100.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 1.6 ),
    MaxInvMass = cms.double( 5.6 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltPsi2SMML3Filtered = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltPsi2SMML2Filtered" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 3.5 ),
    MaxInvMass = cms.double( 3.9 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleMultiMuonNoIso = cms.EDFilter( "HLTPrescaler" )
hltMultiMuonNoIsoLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_TripleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltMultiMuonNoIsoL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMultiMuonNoIsoLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 3 )
)
hltMultiMuonNoIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMultiMuonNoIsoL1Filtered" ),
    MinN = cms.int32( 3 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMultiMuonNoIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMultiMuonNoIsoL2PreFiltered" ),
    MinN = cms.int32( 3 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescalerBLifetime1jet = cms.EDFilter( "HLTPrescaler" )
hltBLifetimeL1seeds = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltBLifetime1jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 180.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltBLifetimeHighestEtJets = cms.EDProducer( "LargestEtCaloJetSelector",
    src = cms.InputTag( "hltIterativeCone5CaloJets" ),
    filter = cms.bool( False ),
    maxNumber = cms.uint32( 4 )
)
hltBLifetimeL25Jets = cms.EDProducer( "EtMinCaloJetSelector",
    src = cms.InputTag( "hltBLifetimeHighestEtJets" ),
    filter = cms.bool( False ),
    etMin = cms.double( 35.0 )
)
hltBLifetimeL25Associator = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltBLifetimeL25Jets" ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltBLifetimeL25TagInfos = cms.EDProducer( "TrackIPProducer",
    jetTracks = cms.InputTag( "hltBLifetimeL25Associator" ),
    primaryVertex = cms.InputTag( "hltPixelVertices" ),
    computeProbabilities = cms.bool( False ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
    minimumNumberOfHits = cms.int32( 3 ),
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    minimumTransverseMomentum = cms.double( 1.0 ),
    maximumDecayLength = cms.double( 5.0 ),
    maximumChiSquared = cms.double( 5.0 ),
    maximumLongitudinalImpactParameter = cms.double( 17.0 ),
    maximumDistanceToJetAxis = cms.double( 0.07 ),
    jetDirectionUsingTracks = cms.bool( False )
)
hltBLifetimeL25BJetTags = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "trackCounting3D2nd" ),
    tagInfos = cms.VInputTag( 'hltBLifetimeL25TagInfos' )
)
hltBLifetimeL25filter = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBLifetimeL25BJetTags" ),
    MinTag = cms.double( 3.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( False )
)
hltBLifetimeL3Jets = cms.EDProducer( "GetJetsFromHLTobject",
    jets = cms.InputTag( "hltBLifetimeL25filter" )
)
hltBLifetimeL3Associator = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltBLifetimeL3Jets" ),
    tracks = cms.InputTag( "hltBLifetimeRegionalCtfWithMaterialTracks" ),
    coneSize = cms.double( 0.5 )
)
hltBLifetimeL3TagInfos = cms.EDProducer( "TrackIPProducer",
    jetTracks = cms.InputTag( "hltBLifetimeL3Associator" ),
    primaryVertex = cms.InputTag( "hltPixelVertices" ),
    computeProbabilities = cms.bool( False ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
    minimumNumberOfHits = cms.int32( 8 ),
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    minimumTransverseMomentum = cms.double( 1.0 ),
    maximumDecayLength = cms.double( 5.0 ),
    maximumChiSquared = cms.double( 5.0 ),
    maximumLongitudinalImpactParameter = cms.double( 17.0 ),
    maximumDistanceToJetAxis = cms.double( 0.07 ),
    jetDirectionUsingTracks = cms.bool( False )
)
hltBLifetimeL3BJetTags = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "trackCounting3D2nd" ),
    tagInfos = cms.VInputTag( 'hltBLifetimeL3TagInfos' )
)
hltBLifetimeL3filter = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBLifetimeL3BJetTags" ),
    MinTag = cms.double( 6.0 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( True )
)
hltPrescalerBLifetime1jet120 = cms.EDFilter( "HLTPrescaler" )
hltBLifetimeL1seedsLowEnergy = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltBLifetime1jetL2filter120 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 120.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltBLifetimeL25JetsRelaxed = cms.EDProducer( "EtMinCaloJetSelector",
    src = cms.InputTag( "hltBLifetimeHighestEtJets" ),
    filter = cms.bool( False ),
    etMin = cms.double( 30.0 )
)
hltBLifetimeL25AssociatorRelaxed = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltBLifetimeL25JetsRelaxed" ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltBLifetimeL25TagInfosRelaxed = cms.EDProducer( "TrackIPProducer",
    jetTracks = cms.InputTag( "hltBLifetimeL25AssociatorRelaxed" ),
    primaryVertex = cms.InputTag( "hltPixelVertices" ),
    computeProbabilities = cms.bool( False ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
    minimumNumberOfHits = cms.int32( 3 ),
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    minimumTransverseMomentum = cms.double( 1.0 ),
    maximumDecayLength = cms.double( 5.0 ),
    maximumChiSquared = cms.double( 5.0 ),
    maximumLongitudinalImpactParameter = cms.double( 17.0 ),
    maximumDistanceToJetAxis = cms.double( 0.07 ),
    jetDirectionUsingTracks = cms.bool( False )
)
hltBLifetimeL25BJetTagsRelaxed = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "trackCounting3D2nd" ),
    tagInfos = cms.VInputTag( 'hltBLifetimeL25TagInfosRelaxed' )
)
hltBLifetimeL25filterRelaxed = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBLifetimeL25BJetTagsRelaxed" ),
    MinTag = cms.double( 2.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( False )
)
hltBLifetimeL3JetsRelaxed = cms.EDProducer( "GetJetsFromHLTobject",
    jets = cms.InputTag( "hltBLifetimeL25filterRelaxed" )
)
hltBLifetimeL3AssociatorRelaxed = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltBLifetimeL3JetsRelaxed" ),
    tracks = cms.InputTag( "hltBLifetimeRegionalCtfWithMaterialTracksRelaxed" ),
    coneSize = cms.double( 0.5 )
)
hltBLifetimeL3TagInfosRelaxed = cms.EDProducer( "TrackIPProducer",
    jetTracks = cms.InputTag( "hltBLifetimeL3AssociatorRelaxed" ),
    primaryVertex = cms.InputTag( "hltPixelVertices" ),
    computeProbabilities = cms.bool( False ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
    minimumNumberOfHits = cms.int32( 8 ),
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    minimumTransverseMomentum = cms.double( 1.0 ),
    maximumDecayLength = cms.double( 5.0 ),
    maximumChiSquared = cms.double( 20.0 ),
    maximumLongitudinalImpactParameter = cms.double( 17.0 ),
    maximumDistanceToJetAxis = cms.double( 0.07 ),
    jetDirectionUsingTracks = cms.bool( False )
)
hltBLifetimeL3BJetTagsRelaxed = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "trackCounting3D2nd" ),
    tagInfos = cms.VInputTag( 'hltBLifetimeL3TagInfosRelaxed' )
)
hltBLifetimeL3filterRelaxed = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBLifetimeL3BJetTagsRelaxed" ),
    MinTag = cms.double( 3.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( True )
)
hltPrescalerBLifetime1jet160 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime1jetL2filter160 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 160.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltPrescalerBLifetime2jet = cms.EDFilter( "HLTPrescaler" )
hltBLifetime2jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 120.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltPrescalerBLifetime2jet60 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime2jetL2filter60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltPrescalerBLifetime2jet100 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime2jetL2filter100 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 100.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltPrescalerBLifetime3jet = cms.EDFilter( "HLTPrescaler" )
hltBLifetime3jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 70.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBLifetime3jet60 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime3jetL2filter60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBLifetime3jet40 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime3jetL2filter40 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBLifetime4jet = cms.EDFilter( "HLTPrescaler" )
hltBLifetime4jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerBLifetime4jet35 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime4jetL2filter35 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerBLifetime4jet30 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime4jetL2filter30 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerBLifetimeHT = cms.EDFilter( "HLTPrescaler" )
hltBLifetimeHTL2filter = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 470.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPrescalerBLifetimeHT420 = cms.EDFilter( "HLTPrescaler" )
hltBLifetimeHTL2filter420 = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 420.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPrescalerBLifetimeHT320 = cms.EDFilter( "HLTPrescaler" )
hltBLifetimeHTL2filter320 = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 320.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPrescalerBSoftmuon2jet = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuonNjetL1seeds = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu5_Jet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltBSoftmuon2jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 120.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltBSoftmuonHighestEtJets = cms.EDProducer( "LargestEtCaloJetSelector",
    src = cms.InputTag( "hltIterativeCone5CaloJets" ),
    filter = cms.bool( False ),
    maxNumber = cms.uint32( 2 )
)
hltBSoftmuonL25Jets = cms.EDProducer( "EtMinCaloJetSelector",
    src = cms.InputTag( "hltBSoftmuonHighestEtJets" ),
    filter = cms.bool( False ),
    etMin = cms.double( 20.0 )
)
hltBSoftmuonL25TagInfos = cms.EDProducer( "SoftLepton",
    jets = cms.InputTag( "hltBSoftmuonL25Jets" ),
    primaryVertex = cms.InputTag( "nominal" ),
    leptons = cms.InputTag( "hltL2Muons" ),
    refineJetAxis = cms.uint32( 0 ),
    leptonDeltaRCut = cms.double( 0.4 ),
    leptonChi2Cut = cms.double( 0.0 ),
    leptonQualityCut = cms.double( 0.0 )
)
hltBSoftmuonL25BJetTags = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "softLeptonByDistance" ),
    tagInfos = cms.VInputTag( 'hltBSoftmuonL25TagInfos' )
)
hltBSoftmuonL25filter = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBSoftmuonL25BJetTags" ),
    MinTag = cms.double( 0.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( False )
)
hltBSoftmuonL3TagInfos = cms.EDProducer( "SoftLepton",
    jets = cms.InputTag( "hltBSoftmuonL25Jets" ),
    primaryVertex = cms.InputTag( "nominal" ),
    leptons = cms.InputTag( "hltL3Muons" ),
    refineJetAxis = cms.uint32( 0 ),
    leptonDeltaRCut = cms.double( 0.4 ),
    leptonChi2Cut = cms.double( 0.0 ),
    leptonQualityCut = cms.double( 0.0 )
)
hltBSoftmuonL3BJetTags = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "softLeptonByPt" ),
    tagInfos = cms.VInputTag( 'hltBSoftmuonL3TagInfos' )
)
hltBSoftmuonL3BJetTagsByDR = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "softLeptonByDistance" ),
    tagInfos = cms.VInputTag( 'hltBSoftmuonL3TagInfos' )
)
hltBSoftmuonL3filter = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBSoftmuonL3BJetTags" ),
    MinTag = cms.double( 0.7 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( True )
)
hltPrescalerBSoftmuon2jet100 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon2jetL2filter100 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 100.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltBSoftmuonL3filterRelaxed = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBSoftmuonL3BJetTags" ),
    MinTag = cms.double( 0.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( True )
)
hltPrescalerBSoftmuon2jet60 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon2jetL2filter60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltPrescalerBSoftmuon3jet = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon3jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 70.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBSoftmuon3jet60 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon3jetL2filter60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBSoftmuon3jet40 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon3jetL2filter40 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBSoftmuon4jet = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon4jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerBSoftmuon4jet35 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon4jetL2filter35 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerBSoftmuon4jet30 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon4jetL2filter30 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerBSoftmuonHT = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuonHTL1seeds = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT300" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltBSoftmuonHTL2filter = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 370.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPrescalerBSoftmuonHT330 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuonHTL2filter330 = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 330.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPrescalerBSoftmuonHT250 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuonHTL1seedsLowEnergy = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT200" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltBSoftmuonHTL2filter250 = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 250.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltJpsitoMumuL1SeedRelaxed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltJpsitoMumuL1FilteredRelaxed = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltJpsitoMumuL1SeedRelaxed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltDisplacedJpsitoMumuFilterRelaxed = cms.EDFilter( "HLTDisplacedmumuFilter",
    Src = cms.InputTag( "hltMuTracks" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinPtPair = cms.double( 4.0 ),
    MinInvMass = cms.double( 1.0 ),
    MaxInvMass = cms.double( 6.0 ),
    ChargeOpt = cms.int32( -1 ),
    FastAccept = cms.bool( False ),
    MinLxySignificance = cms.double( 3.0 ),
    MaxNormalisedChi2 = cms.double( 10.0 ),
    MinCosinePointingAngle = cms.double( 0.9 ),
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" )
)
hltJpsitoMumuL1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltJpsitoMumuL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltJpsitoMumuL1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltDisplacedJpsitoMumuFilter = cms.EDFilter( "HLTDisplacedmumuFilter",
    Src = cms.InputTag( "hltMuTracks" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 4.0 ),
    MinPtPair = cms.double( 4.0 ),
    MinInvMass = cms.double( 1.0 ),
    MaxInvMass = cms.double( 6.0 ),
    ChargeOpt = cms.int32( -1 ),
    FastAccept = cms.bool( False ),
    MinLxySignificance = cms.double( 3.0 ),
    MaxNormalisedChi2 = cms.double( 10.0 ),
    MinCosinePointingAngle = cms.double( 0.9 ),
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" )
)
hltMuMukL1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltMuMukL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuMukL1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltDisplacedMuMukFilter = cms.EDFilter( "HLTDisplacedmumuFilter",
    Src = cms.InputTag( "hltMuTracks" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinPtPair = cms.double( 0.0 ),
    MinInvMass = cms.double( 0.2 ),
    MaxInvMass = cms.double( 3.0 ),
    ChargeOpt = cms.int32( 0 ),
    FastAccept = cms.bool( False ),
    MinLxySignificance = cms.double( 3.0 ),
    MaxNormalisedChi2 = cms.double( 10.0 ),
    MinCosinePointingAngle = cms.double( 0.9 ),
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" )
)
hltmmkFilter = cms.EDFilter( "HLTmmkFilter",
    ThirdTrackMass = cms.double( 0.106 ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinInvMass = cms.double( 1.2 ),
    MaxInvMass = cms.double( 2.2 ),
    MaxNormalisedChi2 = cms.double( 10.0 ),
    MinLxySignificance = cms.double( 3.0 ),
    MinCosinePointingAngle = cms.double( 0.9 ),
    FastAccept = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    MuCand = cms.InputTag( "hltMuTracks" ),
    TrackCand = cms.InputTag( "hltMumukAllConeTracks" )
)
hltSingleTauPrescaler = cms.EDFilter( "HLTPrescaler" )
hltSingleTauL1SeedFilter = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleTauJet80" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltCaloTowersTau1 = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForAll" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 0 )
)
hltIcone5Tau1 = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau1" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltCaloTowersTau2 = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForAll" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 1 )
)
hltIcone5Tau2 = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau2" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltCaloTowersTau3 = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForAll" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 2 )
)
hltIcone5Tau3 = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau3" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltCaloTowersTau4 = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForAll" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 3 )
)
hltIcone5Tau4 = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau4" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hlt1METSingleTau = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 65.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL2SingleTauJets = cms.EDProducer( "L2TauJetsProvider",
    L1ParticlesTau = cms.InputTag( 'l1extraParticles','Tau' ),
    L1ParticlesJet = cms.InputTag( 'l1extraParticles','Central' ),
    L1TauTrigger = cms.InputTag( "hltSingleTauL1SeedFilter" ),
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( 'hltIcone5Tau1','hltIcone5Tau2','hltIcone5Tau3','hltIcone5Tau4' )
)
hltL2SingleTauIsolationProducer = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2SingleTauJets" ),
    EBRecHits = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double( 0.2 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)
hltL2SingleTauIsolationSelector = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2SingleTauIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterSingleTauEcalIsolation = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2SingleTauIsolationSelector','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltAssociatorL25SingleTau = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2SingleTauIsolationSelector','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL25SingleTau = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL25SingleTau" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "offlineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.15 ),
    IsolationCone = cms.double( 0.5 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL25SingleTau = cms.EDProducer( "IsolatedTauJetsSelector",
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    UseIsolationDiscriminator = cms.bool( True ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( 'hltConeIsolationL25SingleTau' )
)
hltFilterL25SingleTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL25SingleTau" ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltAssociatorL3SingleTau = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltIsolatedL25SingleTau" ),
    tracks = cms.InputTag( "hltCtfWithMaterialTracksL3SingleTau" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL3SingleTau = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL3SingleTau" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "offlineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 5 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.15 ),
    IsolationCone = cms.double( 0.5 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL3SingleTau = cms.EDProducer( "IsolatedTauJetsSelector",
    MinimumTransverseMomentumLeadingTrack = cms.double( 20.0 ),
    UseIsolationDiscriminator = cms.bool( False ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( 'hltConeIsolationL3SingleTau' )
)
hltFilterL3SingleTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL3SingleTau" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltSingleTauMETPrescaler = cms.EDFilter( "HLTPrescaler" )
hltSingleTauMETL1SeedFilter = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_TauJet30_ETM30" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hlt1METSingleTauMET = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL2SingleTauMETJets = cms.EDProducer( "L2TauJetsProvider",
    L1ParticlesTau = cms.InputTag( 'l1extraParticles','Tau' ),
    L1ParticlesJet = cms.InputTag( 'l1extraParticles','Central' ),
    L1TauTrigger = cms.InputTag( "hltSingleTauMETL1SeedFilter" ),
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( 'hltIcone5Tau1','hltIcone5Tau2','hltIcone5Tau3','hltIcone5Tau4' )
)
hltL2SingleTauMETIsolationProducer = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2SingleTauMETJets" ),
    EBRecHits = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double( 0.2 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)
hltL2SingleTauMETIsolationSelector = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2SingleTauMETIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterSingleTauMETEcalIsolation = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2SingleTauMETIsolationSelector','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltAssociatorL25SingleTauMET = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2SingleTauMETIsolationSelector','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL25SingleTauMET = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL25SingleTauMET" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "offlineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.15 ),
    IsolationCone = cms.double( 0.5 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL25SingleTauMET = cms.EDProducer( "IsolatedTauJetsSelector",
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    UseIsolationDiscriminator = cms.bool( True ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( 'hltConeIsolationL25SingleTauMET' )
)
hltFilterL25SingleTauMET = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL25SingleTauMET" ),
    MinPt = cms.double( 10.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltAssociatorL3SingleTauMET = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltIsolatedL25SingleTauMET" ),
    tracks = cms.InputTag( "hltCtfWithMaterialTracksL3SingleTauMET" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL3SingleTauMET = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL3SingleTauMET" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "offlineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 5 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.15 ),
    IsolationCone = cms.double( 0.5 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL3SingleTauMET = cms.EDProducer( "IsolatedTauJetsSelector",
    MinimumTransverseMomentumLeadingTrack = cms.double( 15.0 ),
    UseIsolationDiscriminator = cms.bool( False ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( 'hltConeIsolationL3SingleTauMET' )
)
hltFilterL3SingleTauMET = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL3SingleTauMET" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 10.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hlt1METSingleTauRelaxed = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL2SingleTauIsolationSelectorRelaxed = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2SingleTauIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 10000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterSingleTauEcalIsolationRelaxed = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2SingleTauIsolationSelectorRelaxed','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hlt1METSingleTauMETRelaxed = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL2SingleTauMETIsolationSelectorRelaxed = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2SingleTauMETIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterSingleTauMETEcalIsolationRelaxed = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2SingleTauMETIsolationSelectorRelaxed','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltDoubleTauPrescaler = cms.EDFilter( "HLTPrescaler" )
hltDoubleTauL1SeedFilter = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleTauJet40" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltTowerMakerForTaus = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.09 ),
    EEThreshold = cms.double( 0.45 ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.9 ),
    HESThreshold = cms.double( 1.4 ),
    HEDThreshold = cms.double( 1.4 ),
    HOThreshold = cms.double( 1.1 ),
    HF1Threshold = cms.double( 1.2 ),
    HF2Threshold = cms.double( 1.8 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( True ),
    MomConstrMethod = cms.int32( 0 ),
    MomEmDepth = cms.double( 0.0 ),
    MomHadDepth = cms.double( 0.0 ),
    MomTotDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    ecalInputs = cms.VInputTag( 'hltEcalRegionalTausRecHit:EcalRecHitsEB','hltEcalRegionalTausRecHit:EcalRecHitsEE' )
)
hltCaloTowersTau1Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForTaus" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 0 )
)
hltIcone5Tau1Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau1Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltCaloTowersTau2Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForTaus" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 1 )
)
hltIcone5Tau2Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau2Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltCaloTowersTau3Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForTaus" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 2 )
)
hltIcone5Tau3Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau3Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltCaloTowersTau4Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForTaus" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 3 )
)
hltIcone5Tau4Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau4Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltL2DoubleTauJets = cms.EDProducer( "L2TauJetsProvider",
    L1ParticlesTau = cms.InputTag( 'l1extraParticles','Tau' ),
    L1ParticlesJet = cms.InputTag( 'l1extraParticles','Central' ),
    L1TauTrigger = cms.InputTag( "hltDoubleTauL1SeedFilter" ),
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( 'hltIcone5Tau1Regional','hltIcone5Tau2Regional','hltIcone5Tau3Regional','hltIcone5Tau4Regional' )
)
hltL2DoubleTauIsolationProducer = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2DoubleTauJets" ),
    EBRecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double( 0.2 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)
hltL2DoubleTauIsolationSelector = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2DoubleTauIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterDoubleTauEcalIsolation = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2DoubleTauIsolationSelector','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltAssociatorL25PixelTauIsolated = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2DoubleTauIsolationSelector','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL25PixelTauIsolated = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL25PixelTauIsolated" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "offlineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.15 ),
    IsolationCone = cms.double( 0.5 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL25PixelTauPtLeadTk = cms.EDProducer( "IsolatedTauJetsSelector",
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    UseIsolationDiscriminator = cms.bool( False ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( 'hltConeIsolationL25PixelTauIsolated' )
)
hltFilterL25PixelTauPtLeadTk = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL25PixelTauPtLeadTk" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 0.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltIsolatedL25PixelTau = cms.EDProducer( "IsolatedTauJetsSelector",
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    UseIsolationDiscriminator = cms.bool( True ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( 'hltConeIsolationL25PixelTauIsolated' )
)
hltFilterL25PixelTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL25PixelTau" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 0.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltDoubleTauL1SeedFilterRelaxed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleTauJet20" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltL2DoubleTauJetsRelaxed = cms.EDProducer( "L2TauJetsProvider",
    L1ParticlesTau = cms.InputTag( 'l1extraParticles','Tau' ),
    L1ParticlesJet = cms.InputTag( 'l1extraParticles','Central' ),
    L1TauTrigger = cms.InputTag( "hltDoubleTauL1SeedFilterRelaxed" ),
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( 'hltIcone5Tau1Regional','hltIcone5Tau2Regional','hltIcone5Tau3Regional','hltIcone5Tau4Regional' )
)
hltL2DoubleTauIsolationProducerRelaxed = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2DoubleTauJetsRelaxed" ),
    EBRecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double( 0.2 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)
hltL2DoubleTauIsolationSelectorRelaxed = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2DoubleTauIsolationProducerRelaxed','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterDoubleTauEcalIsolationRelaxed = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2DoubleTauIsolationSelectorRelaxed','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltemuPrescale = cms.EDFilter( "HLTPrescaler" )
hltEMuonLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu3_IsoEG5" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltEMuL1MuonFilter = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltEMuonLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 4.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltemuL1IsoSingleL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltEMuonLevel1Seed" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltemuL1IsoSingleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltemuL1IsoSingleL1MatchFilter" ),
    etcut = cms.double( 8.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltemuL1IsoSingleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltemuL1IsoSingleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltEMuL2MuonPreFilter = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltEMuL1MuonFilter" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltEMuL2MuonIsoFilter = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltEMuL2MuonPreFilter" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltemuL1IsoSingleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltemuL1IsoSingleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltemuL1IsoSingleElectronEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltemuL1IsoSingleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 1.5 ),
    eoverpendcapcut = cms.double( 2.45 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltEMuL3MuonPreFilter = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltEMuL2MuonIsoFilter" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltEMuL3MuonIsoFilter = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltEMuL3MuonPreFilter" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltemuL1IsoSingleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltemuL1IsoSingleElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltemuNonIsoPrescale = cms.EDFilter( "HLTPrescaler" )
hltemuNonIsoLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu3_EG12" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltNonIsoEMuL1MuonFilter = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltemuNonIsoLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltemuNonIsoL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltemuNonIsoLevel1Seed" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltemuNonIsoL1IsoEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltemuNonIsoL1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltemuNonIsoL1HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltemuNonIsoL1IsoEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltNonIsoEMuL2MuonPreFilter = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltNonIsoEMuL1MuonFilter" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 8.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltemuNonIsoL1IsoPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltemuNonIsoL1HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltemuNonIsoL1IsoEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltemuNonIsoL1IsoPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    eoverpbarrelcut = cms.double( 1.5 ),
    eoverpendcapcut = cms.double( 2.45 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltNonIsoEMuL3MuonPreFilter = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltNonIsoEMuL2MuonPreFilter" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 10.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltemuNonIsoL1IsoTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltemuNonIsoL1IsoEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltPrescalerElectronTau = cms.EDFilter( "HLTPrescaler" )
hltLevel1GTSeedElectronTau = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_IsoEG10_TauJet20" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltEgammaL1MatchFilterRegionalElectronTau = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltLevel1GTSeedElectronTau" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltEgammaEtFilterElectronTau = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltEgammaL1MatchFilterRegionalElectronTau" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltEgammaHcalIsolFilterElectronTau = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltEgammaEtFilterElectronTau" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltElectronPixelMatchFilterElectronTau = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltEgammaHcalIsolFilterElectronTau" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltElectronOneOEMinusOneOPFilterElectronTau = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltElectronPixelMatchFilterElectronTau" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsL1NonIsoForHLT" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltElectronOneOEMinusOneOPFilterElectronTau" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltL2TauJetsProviderElectronTau = cms.EDProducer( "L2TauJetsProvider",
    L1ParticlesTau = cms.InputTag( 'l1extraParticles','Tau' ),
    L1ParticlesJet = cms.InputTag( 'l1extraParticles','Central' ),
    L1TauTrigger = cms.InputTag( "hltLevel1GTSeedElectronTau" ),
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( 'hltIcone5Tau1Regional','hltIcone5Tau2Regional','hltIcone5Tau3Regional','hltIcone5Tau4Regional' )
)
hltL2ElectronTauIsolationProducer = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2TauJetsProviderElectronTau" ),
    EBRecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double( 0.2 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)
hltL2ElectronTauIsolationSelector = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2ElectronTauIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterEcalIsolatedTauJetsElectronTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2ElectronTauIsolationSelector','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltJetTracksAssociatorAtVertexL25ElectronTau = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2ElectronTauIsolationSelector','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL25ElectronTau = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltJetTracksAssociatorAtVertexL25ElectronTau" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "offlineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.15 ),
    IsolationCone = cms.double( 0.5 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedTauJetsSelectorL25ElectronTauPtLeadTk = cms.EDProducer( "IsolatedTauJetsSelector",
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    UseIsolationDiscriminator = cms.bool( False ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( 'hltConeIsolationL25ElectronTau' )
)
hltFilterIsolatedTauJetsL25ElectronTauPtLeadTk  = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedTauJetsSelectorL25ElectronTauPtLeadTk" ),
    MinPt = cms.double( 0.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltIsolatedTauJetsSelectorL25ElectronTau = cms.EDProducer( "IsolatedTauJetsSelector",
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    UseIsolationDiscriminator = cms.bool( True ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( 'hltConeIsolationL25ElectronTau' )
)
hltFilterIsolatedTauJetsL25ElectronTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedTauJetsSelectorL25ElectronTau" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltElectronBPrescale = cms.EDFilter( "HLTPrescaler" )
hltElectronBL1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_IsoEG10_Jet20" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltElBElectronL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltElectronBL1Seed" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltElBElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltElBElectronL1MatchFilter" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltElBElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltElBElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltElBElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltElBElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltElBElectronEoverpFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltElBElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltElBElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltElBElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltL1seedEJet = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_IsoEG10_Jet30" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltL1IsoEJetSingleL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedEJet" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoEJetSingleEEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoEJetSingleL1MatchFilter" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEJetSingleEHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEJetSingleEPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEJetSingleEEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 2.0 ),
    eoverpendcapcut = cms.double( 2.45 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltL1IsoEJetSingleETrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltej1jet40 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltej2jet80 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 80.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltL1seedEJet30 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_EG5_TripleJet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltL1IsoSingleEJet30L1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedEJet30" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoEJetSingleEEt5Filter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoSingleEJet30L1MatchFilter" ),
    etcut = cms.double( 5.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEJetSingleEEt5HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEEt5Filter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEJetSingleEEt5PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEEt5HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEJetSingleEEt5EoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEEt5PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 2.0 ),
    eoverpendcapcut = cms.double( 2.45 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltL1IsoEJetSingleEEt5TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEEt5EoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltej3jet30 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltej3jet60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltej4jet35 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerMuonTau = cms.EDFilter( "HLTPrescaler" )
hltLevel1GTSeedMuonTau = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu5_TauJet20" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltMuonTauL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltLevel1GTSeedMuonTau" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuonTauIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuonTauL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 12.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuonTauIsoL2IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuonTauIsoL2PreFiltered" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltMuonTauIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuonTauIsoL2IsoFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 14.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuonTauIsoL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuonTauIsoL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltL2TauJetsProviderMuonTau = cms.EDProducer( "L2TauJetsProvider",
    L1ParticlesTau = cms.InputTag( 'l1extraParticles','Tau' ),
    L1ParticlesJet = cms.InputTag( 'l1extraParticles','Central' ),
    L1TauTrigger = cms.InputTag( "hltLevel1GTSeedMuonTau" ),
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( 'hltIcone5Tau1Regional','hltIcone5Tau2Regional','hltIcone5Tau3Regional','hltIcone5Tau4Regional' )
)
hltL2MuonTauIsolationProducer = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2TauJetsProviderMuonTau" ),
    EBRecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double( 0.2 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)
hltL2MuonTauIsolationSelector = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2MuonTauIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterEcalIsolatedTauJetsMuonTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2MuonTauIsolationSelector','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltJetsPixelTracksAssociatorMuonTau = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2MuonTauIsolationSelector','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltPixelTrackConeIsolationMuonTau = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltJetsPixelTracksAssociatorMuonTau" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "offlineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.15 ),
    IsolationCone = cms.double( 0.5 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedTauJetsSelectorL25MuonTauPtLeadTk = cms.EDProducer( "IsolatedTauJetsSelector",
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    UseIsolationDiscriminator = cms.bool( False ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( 'hltPixelTrackConeIsolationMuonTau' )
)
hltFilterL25MuonTauPtLeadTk = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedTauJetsSelectorL25MuonTauPtLeadTk" ),
    MinPt = cms.double( 0.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltPixelTrackIsolatedTauJetsSelectorMuonTau = cms.EDProducer( "IsolatedTauJetsSelector",
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    UseIsolationDiscriminator = cms.bool( True ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( 'hltPixelTrackConeIsolationMuonTau' )
)
hltFilterPixelTrackIsolatedTauJetsMuonTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltPixelTrackIsolatedTauJetsSelectorMuonTau" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 0.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltMuBPrescale = cms.EDFilter( "HLTPrescaler" )
hltMuBLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu5_Jet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltMuBLifetimeL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuBLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 7.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuBLifetimeIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuBLifetimeL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuBLifetimeIsoL2IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuBLifetimeIsoL2PreFiltered" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltMuBLifetimeIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuBLifetimeIsoL2IsoFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuBLifetimeIsoL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuBLifetimeIsoL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltMuBsoftMuPrescale = cms.EDFilter( "HLTPrescaler" )
hltMuBSoftL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuBLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltMuBSoftIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuBSoftL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuBSoftIsoL2IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuBSoftIsoL2PreFiltered" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltMuBSoftIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuBSoftIsoL2IsoFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuBSoftIsoL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuBSoftIsoL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltMuJetsPrescale = cms.EDFilter( "HLTPrescaler" )
hltMuJetsLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu5_Jet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltMuJetsL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuJetsLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 7.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuJetsL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuJetsL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuJetsL2IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuJetsL2PreFiltered" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltMuJetsL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuJetsL2IsoFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuJetsL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuJetsL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltMuJetsHLT1jet40 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltMuNoL2IsoJetsPrescale = cms.EDFilter( "HLTPrescaler" )
hltMuNoL2IsoJetsLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu5_Jet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltMuNoL2IsoJetsL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuNoL2IsoJetsLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 8.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuNoL2IsoJetsL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuNoL2IsoJetsL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 6.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuNoL2IsoJetsL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuNoL2IsoJetsL2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 8.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuNoL2IsoJetsL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuNoL2IsoJetsL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltMuNoL2IsoJetsHLT1jet40 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltMuNoIsoJetsPrescale = cms.EDFilter( "HLTPrescaler" )
hltMuNoIsoJetsLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu5_Jet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltMuNoIsoJetsL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuNoIsoJetsLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 14.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuNoIsoJetsL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuNoIsoJetsL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 11.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuNoIsoJetsL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuNoIsoJetsL2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 14.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltMuNoIsoJetsHLT1jet50 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 50.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltMuNoIsoJets30Level1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu3_TripleJet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltMuNoIsoJetsMinPt4L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuNoIsoJets30Level1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 5.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuNoIsoJetsMinPt4L2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuNoIsoJetsMinPt4L1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuNoIsoJetsMinPtL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuNoIsoJetsMinPt4L2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltMuNoIsoHLTJets3jet30 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBSoftmuon1jet = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon1jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 20.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltBSoftmuonByDRL3filter = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBSoftmuonL3BJetTagsByDR" ),
    MinTag = cms.double( 0.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( True )
)
hltl1sZero = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltpreZero = cms.EDFilter( "HLTPrescaler" )
hltl1sMin = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_MinBias_HTT10" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltpreMin = cms.EDFilter( "HLTPrescaler" )
hltl1sMinHcal = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltpreMinHcal = cms.EDFilter( "HLTPrescaler" )
hltl1sMinEcal = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG2 OR L1_DoubleEG1" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltpreMinEcal = cms.EDFilter( "HLTPrescaler" )
hltPreMinBiasPixel = cms.EDFilter( "HLTPrescaler" )
hltL1seedMinBiasPixel = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPixelCands = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltPixelTracksForMinBias" ),
    particleType = cms.string( "pi+" )
)
hltMinBiasPixelFilter = cms.EDFilter( "HLTPixlMBFilt",
    pixlTag = cms.InputTag( "hltPixelCands" ),
    MinPt = cms.double( 0.0 ),
    MinTrks = cms.uint32( 2 ),
    MinSep = cms.double( 1.0 )
)
hltPreMBForAlignment = cms.EDFilter( "HLTPrescaler" )
hltPixelMBForAlignment = cms.EDFilter( "HLTPixlMBForAlignmentFilter",
    pixlTag = cms.InputTag( "hltPixelCands" ),
    MinPt = cms.double( 5.0 ),
    MinTrks = cms.uint32( 2 ),
    MinSep = cms.double( 1.0 ),
    MinIsol = cms.double( 0.05 )
)
hltLevel1seedHLTBackwardBSC = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "38 OR 39" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrescaleHLTBackwardBSC = cms.EDFilter( "HLTPrescaler" )
hltLevel1seedHLTForwardBSC = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "36 OR 37" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrescaleHLTForwardBSC = cms.EDFilter( "HLTPrescaler" )
hltLevel1seedHLTCSCBeamHalo = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrescaleHLTCSCBeamHalo = cms.EDFilter( "HLTPrescaler" )
hltLevel1seedHLTCSCBeamHaloOverlapRing1 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrescaleHLTCSCBeamHaloOverlapRing1 = cms.EDFilter( "HLTPrescaler" )
hltOverlapsHLTCSCBeamHaloOverlapRing1 = cms.EDFilter( "HLTCSCOverlapFilter",
    input = cms.InputTag( "hltCsc2DRecHits" ),
    minHits = cms.uint32( 4 ),
    xWindow = cms.double( 2.0 ),
    yWindow = cms.double( 2.0 ),
    ring1 = cms.bool( True ),
    ring2 = cms.bool( False ),
    fillHists = cms.bool( False )
)
hltLevel1seedHLTCSCBeamHaloOverlapRing2 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrescaleHLTCSCBeamHaloOverlapRing2 = cms.EDFilter( "HLTPrescaler" )
hltOverlapsHLTCSCBeamHaloOverlapRing2 = cms.EDFilter( "HLTCSCOverlapFilter",
    input = cms.InputTag( "hltCsc2DRecHits" ),
    minHits = cms.uint32( 4 ),
    xWindow = cms.double( 2.0 ),
    yWindow = cms.double( 2.0 ),
    ring1 = cms.bool( False ),
    ring2 = cms.bool( True ),
    fillHists = cms.bool( False )
)
hltLevel1seedHLTCSCBeamHaloRing2or3 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrescaleHLTCSCBeamHaloRing2or3 = cms.EDFilter( "HLTPrescaler" )
hltFilter23HLTCSCBeamHaloRing2or3 = cms.EDFilter( "HLTCSCRing2or3Filter",
    input = cms.InputTag( "hltCsc2DRecHits" ),
    minHits = cms.uint32( 4 ),
    xWindow = cms.double( 2.0 ),
    yWindow = cms.double( 2.0 )
)
hltLevel1seedHLTTrackerCosmics = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "24 OR 25 OR 26 OR 27 OR 28" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPrescaleHLTTrackerCosmics = cms.EDFilter( "HLTPrescaler" )
hltPrescaleTriggerType = cms.EDFilter( "HLTPrescaler" )
hltL1sIsolTrack = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleTauJet30 OR L1_SingleTauJet40 OR L1_SingleTauJet60 OR L1_SingleTauJet80 " ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreIsolTrackNoEcalIso = cms.EDFilter( "HLTPrescaler" )
hltIsolPixelTrackProd = cms.EDProducer( "IsolatedPixelTrackCandidateProducer",
    L1eTauJetsSource = cms.InputTag( 'l1extraParticles','Tau' ),
    tauAssociationCone = cms.double( 0.5 ),
    tauUnbiasCone = cms.double( 1.2 ),
    PixelTracksSource = cms.InputTag( "hltPixelTracks" ),
    PixelIsolationConeSizeHB = cms.double( 0.4 ),
    PixelIsolationConeSizeHE = cms.double( 0.5 ),
    L1GTSeedLabel = cms.InputTag( "hltL1sIsolTrack" ),
    MaxVtxDXYSeed = cms.double( 0.0 ),
    MaxVtxDXYIsol = cms.double( 10.0 ),
    VertexLabel = cms.InputTag( "hltPixelVertices" )
)
hltIsolPixelTrackFilter = cms.EDFilter( "HLTPixelIsolTrackFilter",
    candTag = cms.InputTag( "hltIsolPixelTrackProd" ),
    MinPtTrack = cms.double( 20.0 ),
    MaxPtNearby = cms.double( 2.0 ),
    MaxEtaTrack = cms.double( 1.9 ),
    filterTrackEnergy = cms.bool( False ),
    MinEnergyTrack = cms.double( 15.0 )
)
hltL1sEcalPhiSym = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias OR L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleEG2 OR L1_DoubleEG1 OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltEcalPhiSymPresc = cms.EDFilter( "HLTPrescaler" )
hltAlCaPhiSymStream = cms.EDFilter( "HLTEcalPhiSymFilter",
    barrelHitCollection = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
    endcapHitCollection = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' ),
    phiSymBarrelHitCollection = cms.string( "phiSymEcalRecHitsEB" ),
    phiSymEndcapHitCollection = cms.string( "phiSymEcalRecHitsEE" ),
    eCut_barrel = cms.double( 0.15 ),
    eCut_endcap = cms.double( 0.75 )
)
hltPrePi0Ecal = cms.EDFilter( "HLTPrescaler" )
hltL1sEcalPi0 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleIsoEG5 OR L1_SingleIsoEG8 OR L1_SingleIsoEG10 OR L1_SingleIsoEG12 OR L1_SingleIsoEG15 OR L1_SingleIsoEG20 OR L1_SingleIsoEG25 OR L1_SingleEG5 OR L1_SingleEG8 OR L1_SingleEG10 OR L1_SingleEG12 OR L1_SingleEG15 OR L1_SingleEG20 OR L1_SingleEG25" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltAlCaPi0RegRecHits = cms.EDFilter( "HLTPi0RecHitsFilter",
    barrelHits = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    pi0BarrelHitCollection = cms.string( "pi0EcalRecHitsEB" ),
    gammaCandEtaSize = cms.int32( 9 ),
    gammaCandPhiSize = cms.int32( 21 ),
    clusSeedThr = cms.double( 0.5 ),
    clusEtaSize = cms.int32( 3 ),
    clusPhiSize = cms.int32( 3 ),
    selePtGammaOne = cms.double( 0.9 ),
    selePtGammaTwo = cms.double( 0.9 ),
    selePtPi0 = cms.double( 2.5 ),
    seleMinvMaxPi0 = cms.double( 0.22 ),
    seleMinvMinPi0 = cms.double( 0.06 ),
    seleXtalMinEnergy = cms.double( 0.0 ),
    seleNRHMax = cms.int32( 1000 ),
    seleS4S9GammaOne = cms.double( 0.85 ),
    seleS4S9GammaTwo = cms.double( 0.85 ),
    selePi0Iso = cms.double( 0.5 ),
    selePi0BeltDR = cms.double( 0.2 ),
    selePi0BeltDeta = cms.double( 0.05 ),
    ParameterLogWeighted = cms.bool( True ),
    ParameterX0 = cms.double( 0.89 ),
    ParameterT0_barl = cms.double( 5.7 ),
    ParameterW0 = cms.double( 4.2 ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    l1SeedFilterTag = cms.InputTag( "hltL1sEcalPi0" ),
    ptMinEMObj = cms.double( 5.0 ),
    EMregionEtaMargin = cms.double( 0.25 ),
    EMregionPhiMargin = cms.double( 0.4 )
)
hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
)
hltTriggerSummaryRAWprescaler = cms.EDFilter( "HLTPrescaler" )
hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
hltBoolFinal = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)
hltPixelVertices = cms.EDProducer( "PixelVertexProducer",
    Verbosity = cms.int32( 0 ),
    Finder = cms.string( "DivisiveVertexFinder" ),
    UseError = cms.bool( True ),
    WtAverage = cms.bool( True ),
    ZOffset = cms.double( 5.0 ),
    ZSeparation = cms.double( 0.05 ),
    NTrkMin = cms.int32( 2 ),
    PtMin = cms.double( 1.0 ),
    TrackCollection = cms.InputTag( "hltPixelTracks" ),
    Method2 = cms.bool( True ),
    beamSpot = cms.InputTag( "offlineBeamSpot" )
)
hltPixelMatchElectronsL1IsoLargeWindow = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1IsoLargeWindowWithMaterialTracks" ),
    BSProducer = cms.InputTag( "offlineBeamSpot" )
)
hltPixelMatchElectronsL1NonIsoLargeWindow = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1NonIsoLargeWindowWithMaterialTracks" ),
    BSProducer = cms.InputTag( "offlineBeamSpot" )
)

HLTDoLocalHcalSequence = cms.Sequence( hltHcalDigis + hltHbhereco + hltHfreco + hltHoreco )
HLTDoCaloSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRegionalRestFEDs + hltEcalRegionalRestDigis + hltEcalRegionalRestWeightUncalibRecHit + hltEcalRegionalRestRecHitTmp + hltEcalRecHitAll + hltEcalPreshowerRecHit + HLTDoLocalHcalSequence + hltTowerMakerForAll )
HLTDoJetRecoSequence = cms.Sequence( hltIterativeCone5CaloJets + hltMCJetCorJetIcone5 )
HLTDoHTRecoSequence = cms.Sequence( hltHtMet )
HLTRecoJetMETSequence = cms.Sequence( HLTDoCaloSequence + HLTDoJetRecoSequence + hltMet + HLTDoHTRecoSequence )
HLTRecoJetRegionalSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRegionalJetsFEDs + hltEcalRegionalJetsDigis + hltEcalRegionalJetsWeightUncalibRecHit + hltEcalRegionalJetsRecHitTmp + hltEcalRegionalJetsRecHit + hltEcalPreshowerRecHit + HLTDoLocalHcalSequence + hltTowerMakerForJets + hltIterativeCone5CaloJetsRegional + hltMCJetCorJetIcone5Regional )
HLTDoRegionalEgammaEcalSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRegionalEgammaFEDs + hltEcalRegionalEgammaDigis + hltEcalRegionalEgammaWeightUncalibRecHit + hltEcalRegionalEgammaRecHitTmp + hltEcalRegionalEgammaRecHit + hltEcalPreshowerRecHit )
HLTL1IsolatedEcalClustersSequence = cms.Sequence( hltIslandBasicClustersEndcapL1Isolated + hltIslandBasicClustersBarrelL1Isolated + hltHybridSuperClustersL1Isolated + hltIslandSuperClustersL1Isolated + hltCorrectedIslandEndcapSuperClustersL1Isolated + hltCorrectedIslandBarrelSuperClustersL1Isolated + hltCorrectedHybridSuperClustersL1Isolated + hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated )
HLTDoLocalHcalWithoutHOSequence = cms.Sequence( hltHcalDigis + hltHbhereco + hltHfreco )
HLTPixelMatchElectronL1IsoSequence = cms.Sequence( hltL1IsoElectronPixelSeeds )
HLTPixelMatchElectronL1IsoTrackingSequence = cms.Sequence( hltCkfL1IsoTrackCandidates + hltCtfL1IsoWithMaterialTracks + hltPixelMatchElectronsL1Iso )
HLTL1NonIsolatedEcalClustersSequence = cms.Sequence( hltIslandBasicClustersEndcapL1NonIsolated + hltIslandBasicClustersBarrelL1NonIsolated + hltHybridSuperClustersL1NonIsolated + hltIslandSuperClustersL1NonIsolated + hltCorrectedIslandEndcapSuperClustersL1NonIsolated + hltCorrectedIslandBarrelSuperClustersL1NonIsolated + hltCorrectedHybridSuperClustersL1NonIsolated + hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated )
HLTPixelMatchElectronL1NonIsoSequence = cms.Sequence( hltL1NonIsoElectronPixelSeeds )
HLTPixelMatchElectronL1NonIsoTrackingSequence = cms.Sequence( hltCkfL1NonIsoTrackCandidates + hltCtfL1NonIsoWithMaterialTracks + hltPixelMatchElectronsL1NonIso )
HLTPixelMatchStartUpElectronL1IsoTrackingSequence = cms.Sequence( hltCkfL1IsoStartUpTrackCandidates + hltCtfL1IsoStartUpWithMaterialTracks + hltPixelMatchStartUpElectronsL1Iso )
HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence = cms.Sequence( hltCkfL1NonIsoStartUpTrackCandidates + hltCtfL1NonIsoStartUpWithMaterialTracks + hltPixelMatchStartUpElectronsL1NonIso )
HLTSingleElectronEt12L1NonIsoHLTIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt8 + PreHLT1ElectronLW12L1IHIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTIsoSingleElectronEt12L1MatchFilterRegional + hltL1NonIsoHLTIsoSingleElectronEt12EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTIsoSingleElectronEt12HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTIsoSingleElectronEt12PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTIsoSingleElectronEt12HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTIsoSingleElectronEt12TrackIsolFilter )
HLTPixelMatchElectronL1IsoLargeWindowSequence = cms.Sequence( hltL1IsoLargeWindowElectronPixelSeeds )
HLTPixelMatchElectronL1NonIsoLargeWindowSequence = cms.Sequence( hltL1NonIsoLargeWindowElectronPixelSeeds )
HLTSingleElectronEt15L1NonIsoHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt12 + PreHLT1Electron15L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSingleElectronEt15L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSingleElectronEt15EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTLooseIsoSingleElectronEt15HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTLooseIsoSingleElectronEt15PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTLooseIsoSingleElectronEt15HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTLooseIsoSingleElectronEt15TrackIsolFilter )
HLTSingleElectronLWEt18L1NonIsoHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt15 + PreHLT1ElectronLWEt18L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSingleElectronLWEt18L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSingleElectronLWEt18EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTLooseIsoSingleElectronLWEt18PixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HOneOEMinusOneOPFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowElectronTrackIsol + hltL1NonIsoHLTLooseIsoSingleElectronLWEt18TrackIsolFilter )
HLTSingleElectronLWEt15L1NonIsoHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt12 + PreHLT1ElectronLWEt15L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSingleElectronLWEt15L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSingleElectronLWEt15EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTLooseIsoSingleElectronLWEt15PixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HOneOEMinusOneOPFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowElectronTrackIsol + hltL1NonIsoHLTLooseIsoSingleElectronLWEt15TrackIsolFilter )
HLTSingleElectronEt8L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltSingleElectronEt8L1NonIsoHLTnoIsoPresc + hltL1seedRelaxedSingleEgammaEt5 + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTnoIsoSingleElectronEt8L1MatchFilterRegional + hltL1NonIsoHLTnoIsoSingleElectronEt8EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTnoIsoSingleElectronEt8HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTnoIsoSingleElectronEt8PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTnoIsoSingleElectronEt8HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTnoIsoSingleElectronEt8TrackIsolFilter )
HLTSingleElectronEt10L1NonIsoHLTnonIsoSequence = cms.Sequence( HLTBeginSequence + hltSingleElectronEt10L1NonIsoHLTNonIsoPresc + hltL1seedRelaxedSingleEgammaEt8 + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSingleElectronEt10L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSingleElectronEt10EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonHLTnonIsoIsoSingleElectronEt10HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTnonIsoSingleElectronEt10HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTNonIsoSingleElectronEt10TrackIsolFilter )
HLTSingleElectronEt15L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt12 + PreHLT1Electron15L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSingleElectronEt15L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSingleElectronEt15EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoSingleElectronEt15HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTNonIsoSingleElectronEt15HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter )
HLTSingleElectronEt18L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt15 + PreHLT1Electron18L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSingleElectronEt18L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSingleElectronEt18EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoSingleElectronEt18HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTNonIsoSingleElectronEt18PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTNonIsoSingleElectronEt18HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTNonIsoSingleElectronEt18TrackIsolFilter )
HLTSingleElectronLWEt12L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt8 + PreHLT1ElectronLWEt12L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSingleElectronLWEt12L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSingleElectronLWEt12EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoSingleElectronLWEt12HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTNonIsoSingleElectronLWEt12PixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + hltL1NonIsoHLTNonIsoSingleElectronLWEt12HOneOEMinusOneOPFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowElectronTrackIsol + hltL1NonIsoHLTNonIsoSingleElectronLWEt12TrackIsolFilter )
HLTSingleElectronLWEt15L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt10 + PreHLT1ElectronLWEt15L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSingleElectronLWEt15L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoSingleElectronLWEt15HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + hltL1NonIsoHLTNonIsoSingleElectronLWEt15HOneOEMinusOneOPFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowElectronTrackIsol + hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter )
HLTDoLocalTrackerSequence = cms.Sequence( HLTDoLocalPixelSequence + HLTDoLocalStripSequence )
HLTDoubleElectronEt5L1NonIsoHLTnonIsoSequence = cms.Sequence( HLTBeginSequence + hltDoubleElectronEt5L1NonIsoHLTNonIsoPresc + hltL1seedRelaxedDoubleEgammaEt5 + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonHLTnonIsoIsoDoubleElectronEt5HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTnonIsoDoubleElectronEt5HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter )
HLTDoubleElectronLWonlyPMEt8L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEgammaEt5 + PreHLT2ElectronLWonlyPMEt8L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8PixelMatchFilter )
HLTDoubleElectronLWonlyPMEt10L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEgammaEt5 + PreHLT2ElectronLWonlyPMEt10L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10PixelMatchFilter )
HLTDoubleElectronLWonlyPMEt12L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEgammaEt10 + PreHLT2ElectronLWonlyPMEt12L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12PixelMatchFilter )
HLTSinglePhotonEt10L1NonIsolatedSequence = cms.Sequence( HLTBeginSequence + hltSinglePhotonEt10L1NonIsoPresc + hltL1seedRelaxedSingleEgammaEt8 + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoSinglePhotonEt10L1MatchFilterRegional + hltL1NonIsoSinglePhotonEt10EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoSinglePhotonEt10EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoSinglePhotonEt10HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoSinglePhotonEt10TrackIsolFilter )
HLTSinglePhoton15L1NonIsolatedHLTIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt12 + PreHLT1Photon15L1RHIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTIsoSinglePhotonEt15L1MatchFilterRegional + hltL1NonIsoHLTIsoSinglePhotonEt15EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTIsoSinglePhotonEt15EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTIsoSinglePhotonEt15HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTIsoSinglePhotonEt15TrackIsolFilter )
HLTSinglePhoton20L1NonIsolatedHLTIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt15 + PreHLT1Photon20L1RHIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTIsoSinglePhotonEt20L1MatchFilterRegional + hltL1NonIsoHLTIsoSinglePhotonEt20EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTIsoSinglePhotonEt20EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTIsoSinglePhotonEt20HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTIsoSinglePhotonEt20TrackIsolFilter )
HLTSinglePhoton25L1NonIsolatedHLTIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt15 + PreHLT1Photon25L1RHIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTIsoSinglePhotonEt25L1MatchFilterRegional + hltL1NonIsoHLTIsoSinglePhotonEt25EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTIsoSinglePhotonEt25EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTIsoSinglePhotonEt25HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTIsoSinglePhotonEt25TrackIsolFilter )
HLTSinglePhoton20L1NonIsolatedHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt15 + PreHLT1Photon20L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSinglePhotonEt20L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSinglePhotonEt20EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt20EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt20HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt20TrackIsolFilter )
HLTSinglePhoton30L1NonIsolatedHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt15 + PreHLT1Photon30L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSinglePhotonEt30L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSinglePhotonEt30EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt30EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt30HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt30TrackIsolFilter )
HLTSinglePhoton40L1NonIsolatedHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt15 + PreHLT1Photon40L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSinglePhotonEt40L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSinglePhotonEt40EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt40EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt40HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt40TrackIsolFilter )
HLTSinglePhoton45L1NonIsolatedHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt15 + PreHLT1Photon45L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSinglePhotonEt45L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSinglePhotonEt45EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt45EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt45HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt45TrackIsolFilter )
HLTSinglePhoton15L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt10 + PreHLT1Photon15L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter )
HLTSinglePhoton25L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt15 + PreHLT1Photon25L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSinglePhotonEt25L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSinglePhotonEt25EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt25EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt25TrackIsolFilter )
HLTSinglePhoton30L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt15 + PreHLT1Photon30L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSinglePhotonEt30L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSinglePhotonEt30EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt30EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt30TrackIsolFilter )
HLTSinglePhoton40L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEgammaEt15 + PreHLT1Photon40L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSinglePhotonEt40L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSinglePhotonEt40EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt40EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt40HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt40TrackIsolFilter )
HLTDoublePhoton20L1NonIsolatedHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEgammaEt10 + PreHLT2Photon20L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoDoublePhotonEt20L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoDoublePhotonEt20EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTLooseIsoDoublePhotonEt20EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTLooseIsoDoublePhotonEt20HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTLooseIsoDoublePhotonEt20TrackIsolFilter )
HLTDoublePhoton8L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEgammaEt5 + PreHLT2Photon8L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoublePhotonEt8L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoublePhotonEt8EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt8EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt8HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt8TrackIsolFilter )
HLTDoublePhoton10L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEgammaEt5 + PreHLT2Photon10L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoublePhotonEt10L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoublePhotonEt10EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt10EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt10TrackIsolFilter )
HLTDoublePhoton20L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEgammaEt10 + PreHLT2Photon20L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoublePhotonEt20L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoublePhotonEt20EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt20EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt20HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt20TrackIsolFilter )
HLTL1muonrecoSequence = cms.Sequence( HLTBeginSequence )
HLTL2muonrecoNocandSequence = cms.Sequence( cms.SequencePlaceholder("simMuonDTDigis") + hltDt1DRecHits + hltDt4DSegments + cms.SequencePlaceholder("simMuonCSCDigis") + hltCsc2DRecHits + hltCscSegments + cms.SequencePlaceholder("simMuonRPCDigis") + hltRpcRecHits + hltL2MuonSeeds + hltL2Muons )
HLTL2muonrecoSequence = cms.Sequence( HLTL2muonrecoNocandSequence + hltL2MuonCandidates )
HLTL2muonisorecoSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRegionalMuonsFEDs + hltEcalRegionalMuonsDigis + hltEcalRegionalMuonsWeightUncalibRecHit + hltEcalRegionalMuonsRecHitTmp + hltEcalRegionalMuonsRecHit + hltEcalPreshowerRecHit + HLTDoLocalHcalSequence + hltTowerMakerForMuons + hltL2MuonIsolations )
HLTL3muonTkCandidateSequence = cms.Sequence( HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL3TrajectorySeed + hltL3TrackCandidateFromL2 )
HLTL3muonrecoNocandSequence = cms.Sequence( HLTL3muonTkCandidateSequence + hltL3Muons )
HLTL3muonrecoSequence = cms.Sequence( HLTL3muonrecoNocandSequence + hltL3MuonCandidates )
HLTL3muonisorecoSequence = cms.Sequence( hltPixelTracks + hltL3MuonIsolations )
HLTL3TkmuonrecoNocandSequence = cms.Sequence( HLTL3muonTkCandidateSequence + hltL3TkMuons )
HLTBCommonL2recoSequence = cms.Sequence( HLTDoCaloSequence + HLTDoJetRecoSequence + HLTDoHTRecoSequence )
HLTBLifetimeL25recoSequence = cms.Sequence( hltBLifetimeHighestEtJets + hltBLifetimeL25Jets + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltBLifetimeL25Associator + hltBLifetimeL25TagInfos + hltBLifetimeL25BJetTags )
HLTBLifetimeL3recoSequence = cms.Sequence( hltBLifetimeL3Jets + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltBLifetimeRegionalPixelSeedGenerator + hltBLifetimeRegionalCkfTrackCandidates + hltBLifetimeRegionalCtfWithMaterialTracks + hltBLifetimeL3Associator + hltBLifetimeL3TagInfos + hltBLifetimeL3BJetTags )
HLTBLifetimeL25recoSequenceRelaxed = cms.Sequence( hltBLifetimeHighestEtJets + hltBLifetimeL25JetsRelaxed + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltBLifetimeL25AssociatorRelaxed + hltBLifetimeL25TagInfosRelaxed + hltBLifetimeL25BJetTagsRelaxed )
HLTBLifetimeL3recoSequenceRelaxed = cms.Sequence( hltBLifetimeL3JetsRelaxed + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltBLifetimeRegionalPixelSeedGeneratorRelaxed + hltBLifetimeRegionalCkfTrackCandidatesRelaxed + hltBLifetimeRegionalCtfWithMaterialTracksRelaxed + hltBLifetimeL3AssociatorRelaxed + hltBLifetimeL3TagInfosRelaxed + hltBLifetimeL3BJetTagsRelaxed )
HLTBSoftmuonL25recoSequence = cms.Sequence( hltBSoftmuonHighestEtJets + hltBSoftmuonL25Jets + HLTL2muonrecoNocandSequence + hltBSoftmuonL25TagInfos + hltBSoftmuonL25BJetTags )
HLTBSoftmuonL3recoSequence = cms.Sequence( HLTL3muonrecoNocandSequence + hltBSoftmuonL3TagInfos + hltBSoftmuonL3BJetTags + hltBSoftmuonL3BJetTagsByDR )
HLTL3displacedMumurecoSequence = cms.Sequence( HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTRecopixelvertexingSequence + hltMumuPixelSeedFromL2Candidate + hltCkfTrackCandidatesMumu + hltCtfWithMaterialTracksMumu + hltMuTracks )
HLTCaloTausCreatorSequence = cms.Sequence( HLTDoCaloSequence + hltCaloTowersTau1 + hltIcone5Tau1 + hltCaloTowersTau2 + hltIcone5Tau2 + hltCaloTowersTau3 + hltIcone5Tau3 + hltCaloTowersTau4 + hltIcone5Tau4 )
HLTCaloTausCreatorRegionalSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRegionalTausFEDs + hltEcalRegionalTausDigis + hltEcalRegionalTausWeightUncalibRecHit + hltEcalRegionalTausRecHitTmp + hltEcalRegionalTausRecHit + hltEcalPreshowerRecHit + HLTDoLocalHcalSequence + hltTowerMakerForTaus + hltCaloTowersTau1Regional + hltIcone5Tau1Regional + hltCaloTowersTau2Regional + hltIcone5Tau2Regional + hltCaloTowersTau3Regional + hltIcone5Tau3Regional + hltCaloTowersTau4Regional + hltIcone5Tau4Regional )
HLTETauSingleElectronL1IsolatedHOneOEMinusOneOPFilterSequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltEgammaL1MatchFilterRegionalElectronTau + hltEgammaEtFilterElectronTau + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltEgammaHcalIsolFilterElectronTau + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltElectronPixelMatchFilterElectronTau + HLTPixelMatchElectronL1IsoTrackingSequence + hltElectronOneOEMinusOneOPFilterElectronTau + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau )
HLTL2TauJetsElectronTauSequnce = cms.Sequence( HLTCaloTausCreatorRegionalSequence + hltL2TauJetsProviderElectronTau )
HLTL1EplusJetSequence = cms.Sequence( HLTBeginSequence + hltL1seedEJet )
HLTEJetElectronSequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoEJetSingleL1MatchFilter + hltL1IsoEJetSingleEEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoEJetSingleEHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltL1IsoEJetSingleEPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltL1IsoEJetSingleEEoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1IsoEJetSingleETrackIsolFilter + hltSingleElectronL1IsoPresc )
HLTL1EplusJet30Sequence = cms.Sequence( HLTBeginSequence + hltL1seedEJet30 )
HLTE3Jet30ElectronSequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoSingleEJet30L1MatchFilter + hltL1IsoEJetSingleEEt5Filter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoEJetSingleEEt5HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltL1IsoEJetSingleEEt5PixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltL1IsoEJetSingleEEt5EoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1IsoEJetSingleEEt5TrackIsolFilter + hltSingleElectronL1IsoPresc )
HLTL1SeedFilterSequence = cms.Sequence( hltL1sIsolTrack )
HLTL3PixelIsolFilterSequence = cms.Sequence( HLTDoLocalPixelSequence + hltPixelTracking + hltPixelVertices + hltIsolPixelTrackProd + hltIsolPixelTrackFilter )
HLTIsoTrRegFEDSelection = cms.Sequence( hltSiStripRegFED + hltEcalRegFED + hltSubdetFED )

HLTriggerFirstPath = cms.Path( HLTBeginSequence + hltBoolFirst + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1Jet15 = cms.Path( HLTBeginSequence + hltPre1Level1jet15 + hltL1s1Level1jet15 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet30 = cms.Path( HLTBeginSequence + hltL1s1jet30 + hltPre1jet30 + HLTRecoJetMETSequence + hlt1jet30 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet50 = cms.Path( HLTBeginSequence + hltL1s1jet50 + hltPre1jet50 + HLTRecoJetMETSequence + hlt1jet50 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet80 = cms.Path( HLTBeginSequence + hltL1s1jet80 + hltPre1jet80 + HLTRecoJetRegionalSequence + hlt1jet80 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet110 = cms.Path( HLTBeginSequence + hltL1s1jet110 + hltPre1jet110 + HLTRecoJetRegionalSequence + hlt1jet110 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet180 = cms.Path( HLTBeginSequence + hltL1s1jet180 + hltPre1jet180 + HLTRecoJetRegionalSequence + hlt1jet180regional + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet250 = cms.Path( HLTBeginSequence + hltL1s1jet250 + hltPre1jet250 + HLTRecoJetRegionalSequence + hlt1jet250 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_FwdJet20 = cms.Path( HLTBeginSequence + hltL1RapGap + hltPrerapgap + HLTRecoJetMETSequence + hltRapGap + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleJet150 = cms.Path( HLTBeginSequence + hltL1s2jet + hltPre2jet + HLTRecoJetRegionalSequence + hlt2jet150 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleJet125_Aco = cms.Path( HLTBeginSequence + hltL1s2jetAco + hltPre2jetAco + HLTRecoJetRegionalSequence + hlt2jet125 + hlt2jetAco + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleFwdJet50 = cms.Path( HLTBeginSequence + hltL1s2jetGap + hltPre2jetGap + HLTRecoJetMETSequence + hlt2jetGapFilter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DiJetAve15 = cms.Path( HLTBeginSequence + hltL1sdijetave15 + hltPredijetave15 + HLTRecoJetMETSequence + hltdijetave15 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DiJetAve30 = cms.Path( HLTBeginSequence + hltL1sdijetave30 + hltPredijetave30 + HLTRecoJetMETSequence + hltdijetave30 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DiJetAve50 = cms.Path( HLTBeginSequence + hltL1sdijetave50 + hltPredijetave50 + HLTRecoJetMETSequence + hltdijetave50 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DiJetAve70 = cms.Path( HLTBeginSequence + hltL1sdijetave70 + hltPredijetave70 + HLTRecoJetMETSequence + hltdijetave70 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DiJetAve130 = cms.Path( HLTBeginSequence + hltL1sdijetave130 + hltPredijetave130 + HLTRecoJetMETSequence + hltdijetave130 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DiJetAve220 = cms.Path( HLTBeginSequence + hltL1sdijetave220 + hltPredijetave220 + HLTRecoJetMETSequence + hltdijetave220 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_TripleJet85 = cms.Path( HLTBeginSequence + hltL1s3jet + hltPre3jet + HLTRecoJetRegionalSequence + hlt3jet85 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_QuadJet30 = cms.Path( HLTBeginSequence + hltL1s4jet30 + hltPre4jet30 + HLTRecoJetMETSequence + hlt4jet30 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_QuadJet60 = cms.Path( HLTBeginSequence + hltL1s4jet + hltPre4jet + HLTRecoJetMETSequence + hlt4jet60 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_SumET120 = cms.Path( HLTBeginSequence + hltL1s1SumET + hltPre1MET1SumET + HLTRecoJetMETSequence + hlt1SumET120 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1MET20 = cms.Path( HLTBeginSequence + hltPreLevel1MET20 + hltL1sLevel1MET20 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MET25 = cms.Path( HLTBeginSequence + hltL1s1MET25 + hltPre1MET25 + HLTRecoJetMETSequence + hlt1MET25 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MET35 = cms.Path( HLTBeginSequence + hltL1s1MET35 + hltPre1MET35 + HLTRecoJetMETSequence + hlt1MET35 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MET50 = cms.Path( HLTBeginSequence + hltL1s1MET50 + hltPre1MET50 + HLTRecoJetMETSequence + hlt1MET50 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MET65 = cms.Path( HLTBeginSequence + hltL1s1MET65 + hltPre1MET65 + HLTRecoJetMETSequence + hlt1MET65 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MET75 = cms.Path( HLTBeginSequence + hltL1s1MET75 + hltPre1MET75 + HLTRecoJetMETSequence + hlt1MET75 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MET35_HT350 = cms.Path( HLTBeginSequence + hltL1s1MET1HT + hltPre1MET1HT + HLTRecoJetMETSequence + hlt1MET65 + hlt1HT350 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet180_MET60 = cms.Path( HLTBeginSequence + hltL1s1jet1MET + hltPre1jet1MET + HLTRecoJetMETSequence + hlt1MET60 + hlt1jet180 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet60_MET70_Aco = cms.Path( HLTBeginSequence + hltL1sPhiJet1MET + hltPrephijet1met + HLTRecoJetMETSequence + hlt1MET70 + hltPhiJet1metAco + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet100_MET60_Aco = cms.Path( HLTBeginSequence + hltL1s1jet1METAco + hltPre1jet1METAco + HLTRecoJetMETSequence + hlt1MET60 + hlt1jet100 + hlt1jet1METAco + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleJet125_MET60 = cms.Path( HLTBeginSequence + hltL1s2jet1MET + hltPre2jet1MET + HLTRecoJetMETSequence + hlt1MET60 + hlt2jet125New + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleFwdJet40_MET60 = cms.Path( HLTBeginSequence + hltL1s2jetvbfMET + hltPre2jetvbfMET + HLTRecoJetMETSequence + hlt1MET60 + hlt2jetvbf + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleJet60_MET60_Aco = cms.Path( HLTBeginSequence + hltL1sPhi2MET + hltPrephi2met + HLTRecoJetMETSequence + hlt1MET60 + hltPhi2metAco + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleJet50_MET70_Aco = cms.Path( HLTBeginSequence + hltL1sPhiJet2MET + hltPrephijet2met + HLTRecoJetMETSequence + hlt1MET70 + hltPhiJet2metAco + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleJet40_MET70_Aco = cms.Path( HLTBeginSequence + hltL1sPhiJet1Jet2 + hltPrephijet1jet2 + HLTRecoJetMETSequence + hlt1MET70 + hltPhiJet1Jet2Aco + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet80_Jet20_MET60_NV = cms.Path( HLTBeginSequence + hltL1snvMET + hltPrenv + HLTRecoJetMETSequence + hlt1MET60 + hltnv + cms.SequencePlaceholder("HLTEndSequence") )
HLT_TripleJet60_MET60 = cms.Path( HLTBeginSequence + hltL1s3jet1MET + hltPre3jet1MET + HLTRecoJetMETSequence + hlt1MET60 + hlt3jet60 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_QuadJet35_MET60 = cms.Path( HLTBeginSequence + hltL1s4jet1MET + hltPre4jet1MET + HLTRecoJetMETSequence + hlt1MET60 + hlt4jet35 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle15_L1I = cms.Path( HLTBeginSequence + hltL1seedSingleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoSingleL1MatchFilter + hltL1IsoSingleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoSingleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltL1IsoSingleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltL1IsoSingleElectronHOneOEMinusOneOPFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1IsoSingleElectronTrackIsolFilter + hltSingleElectronL1IsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle18_L1R = cms.Path( HLTBeginSequence + hltL1seedRelaxedSingleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoSingleElectronL1MatchFilterRegional + hltL1NonIsoSingleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoSingleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + HLTPixelMatchElectronL1NonIsoSequence + hltL1NonIsoSingleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoTrackingSequence + hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + HLTL1NonIsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1NonIsoElectronTrackIsol + hltL1NonIsoSingleElectronTrackIsolFilter + hltSingleElectronL1NonIsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle12_SW_L1R = cms.Path( HLTSingleElectronEt12L1NonIsoHLTIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle15_LW_L1I = cms.Path( HLTBeginSequence + hltL1seedSingleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoLargeWindowSingleL1MatchFilter + hltL1IsoLargeWindowSingleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoLargeWindowSingleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoLargeWindowSequence + hltL1IsoLargeWindowSingleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1IsoLargeWindowSingleElectronTrackIsolFilter + hltSingleElectronL1IsoLargeWindowPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle18_LW_L1R = cms.Path( HLTBeginSequence + hltL1seedRelaxedSingleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional + hltL1NonIsoLargeWindowSingleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoLargeWindowSequence + HLTPixelMatchElectronL1NonIsoLargeWindowSequence + hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter + hltSingleElectronL1NonIsoLargeWindowPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_LooseIsoEle15_SW_L1R = cms.Path( HLTSingleElectronEt15L1NonIsoHLTLooseIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_LooseIsoEle18_LW_L1R = cms.Path( HLTSingleElectronLWEt18L1NonIsoHLTLooseIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_LooseIsoEle15_LW_L1R = cms.Path( HLTSingleElectronLWEt15L1NonIsoHLTLooseIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Ele8_SW_L1R = cms.Path( HLTSingleElectronEt8L1NonIsoHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Ele10_SW_L1R = cms.Path( HLTSingleElectronEt10L1NonIsoHLTnonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Ele15_SW_L1R = cms.Path( HLTSingleElectronEt15L1NonIsoHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Ele18_SW_L1R = cms.Path( HLTSingleElectronEt18L1NonIsoHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Ele12_LW_L1R = cms.Path( HLTSingleElectronLWEt12L1NonIsoHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Ele15_LW_L1R = cms.Path( HLTSingleElectronLWEt15L1NonIsoHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_EM80 = cms.Path( HLTBeginSequence + hltL1seedRelaxedSingleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoSingleEMHighEtL1MatchFilterRegional + hltL1NonIsoSinglePhotonEMHighEtEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoSingleEMHighEtEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1NonIsolatedElectronHcalIsol + hltL1IsolatedElectronHcalIsol + hltL1NonIsoSingleEMHighEtHOEFilter + hltHcalDoubleCone + hltL1NonIsoEMHcalDoubleCone + hltL1NonIsoSingleEMHighEtHcalDBCFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoSingleEMHighEtTrackIsolFilter + hltSingleEMVHighEtL1NonIsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_EM200 = cms.Path( HLTBeginSequence + hltL1seedRelaxedSingleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional + hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter + hltSingleEMVHEL1NonIsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleIsoEle10_L1I = cms.Path( HLTBeginSequence + hltL1seedDoubleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoDoubleElectronL1MatchFilterRegional + hltL1IsoDoubleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoDoubleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltL1IsoDoubleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltL1IsoDoubleElectronEoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1IsoDoubleElectronTrackIsolFilter + hltDoubleElectronL1IsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleIsoEle12_L1R = cms.Path( HLTBeginSequence + hltL1seedRelaxedDoubleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoDoubleElectronL1MatchFilterRegional + hltL1NonIsoDoubleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoDoubleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + HLTPixelMatchElectronL1NonIsoSequence + hltL1NonIsoDoubleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoTrackingSequence + hltL1NonIsoDoubleElectronEoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + HLTL1NonIsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1NonIsoElectronTrackIsol + hltL1NonIsoDoubleElectronTrackIsolFilter + hltDoubleElectronL1NonIsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleIsoEle10_LW_L1I = cms.Path( HLTBeginSequence + hltL1seedDoubleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional + hltL1IsoLargeWindowDoubleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoLargeWindowDoubleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoLargeWindowSequence + hltL1IsoLargeWindowDoubleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + hltL1IsoLargeWindowDoubleElectronEoverpFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1IsoLargeWindowDoubleElectronTrackIsolFilter + hltDoubleElectronL1IsoLargeWindowPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleIsoEle12_LW_L1R = cms.Path( HLTBeginSequence + hltL1seedRelaxedDoubleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional + hltL1NonIsoLargeWindowDoubleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoLargeWindowSequence + HLTPixelMatchElectronL1NonIsoLargeWindowSequence + hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + hltL1NonIsoLargeWindowDoubleElectronEoverpFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter + hltDoubleElectronL1NonIsoLargeWindowPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleEle5_SW_L1R = cms.Path( HLTDoubleElectronEt5L1NonIsoHLTnonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleEle8_LW_OnlyPixelM_L1R = cms.Path( HLTDoubleElectronLWonlyPMEt8L1NonIsoHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleEle10_LW_OnlyPixelM_L1R = cms.Path( HLTDoubleElectronLWonlyPMEt10L1NonIsoHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleEle12_LW_OnlyPixelM_L1R = cms.Path( HLTDoubleElectronLWonlyPMEt12L1NonIsoHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleEle10_Z = cms.Path( HLTBeginSequence + hltL1seedDoubleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoDoubleElectronZeeL1MatchFilterRegional + hltL1IsoDoubleElectronZeeEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoDoubleElectronZeeHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltL1IsoDoubleElectronZeePixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltL1IsoDoubleElectronZeeEoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1IsoDoubleElectronZeeTrackIsolFilter + hltL1IsoDoubleElectronZeePMMassFilter + hltZeeCounterPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleEle6_Exclusive = cms.Path( HLTBeginSequence + hltL1seedExclusiveDoubleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoDoubleExclElectronL1MatchFilterRegional + hltL1IsoDoubleExclElectronEtPhiFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoDoubleExclElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltL1IsoDoubleExclElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltL1IsoDoubleExclElectronEoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1IsoDoubleExclElectronTrackIsolFilter + hltDoubleExclElectronL1IsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoPhoton12_L1I = cms.Path( HLTBeginSequence + hltL1seedSingleEgammaPrescaled + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoSinglePhotonPrescaledL1MatchFilter + hltL1IsoSinglePhotonPrescaledEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1IsoSinglePhotonPrescaledEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1IsoSinglePhotonPrescaledHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1IsoSinglePhotonPrescaledTrackIsolFilter + hltSinglePhotonPrescaledL1IsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoPhoton30_L1I = cms.Path( HLTBeginSequence + hltL1seedSingleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoSinglePhotonL1MatchFilter + hltL1IsoSinglePhotonEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1IsoSinglePhotonEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1IsoSinglePhotonHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1IsoSinglePhotonTrackIsolFilter + hltSinglePhotonL1IsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoPhoton10_L1R = cms.Path( HLTSinglePhotonEt10L1NonIsolatedSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoPhoton15_L1R = cms.Path( HLTSinglePhoton15L1NonIsolatedHLTIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoPhoton20_L1R = cms.Path( HLTSinglePhoton20L1NonIsolatedHLTIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoPhoton25_L1R = cms.Path( HLTSinglePhoton25L1NonIsolatedHLTIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoPhoton40_L1R = cms.Path( HLTBeginSequence + hltL1seedRelaxedSingleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoSinglePhotonL1MatchFilterRegional + hltL1NonIsoSinglePhotonEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoSinglePhotonEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoSinglePhotonHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoSinglePhotonTrackIsolFilter + hltSinglePhotonL1NonIsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_LooseIsoPhoton20_L1R = cms.Path( HLTSinglePhoton20L1NonIsolatedHLTLooseIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_LooseIsoPhoton30_L1R = cms.Path( HLTSinglePhoton30L1NonIsolatedHLTLooseIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_LooseIsoPhoton40_L1R = cms.Path( HLTSinglePhoton40L1NonIsolatedHLTLooseIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_LooseIsoPhoton45_L1R = cms.Path( HLTSinglePhoton45L1NonIsolatedHLTLooseIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Photon15_L1R = cms.Path( HLTSinglePhoton15L1NonIsolatedHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Photon25_L1R = cms.Path( HLTSinglePhoton25L1NonIsolatedHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Photon30_L1R = cms.Path( HLTSinglePhoton30L1NonIsolatedHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Photon40_L1R = cms.Path( HLTSinglePhoton40L1NonIsolatedHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleIsoPhoton20_L1I = cms.Path( HLTBeginSequence + hltL1seedDoubleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoDoublePhotonL1MatchFilterRegional + hltL1IsoDoublePhotonEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1IsoDoublePhotonEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1IsoDoublePhotonHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1IsoDoublePhotonTrackIsolFilter + hltL1IsoDoublePhotonDoubleEtFilter + hltDoublePhotonL1IsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleIsoPhoton20_L1R = cms.Path( HLTBeginSequence + hltL1seedRelaxedDoubleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoDoublePhotonL1MatchFilterRegional + hltL1NonIsoDoublePhotonEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoDoublePhotonEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoDoublePhotonHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoDoublePhotonTrackIsolFilter + hltL1NonIsoDoublePhotonDoubleEtFilter + hltDoublePhotonL1NonIsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleLooseIsoPhoton20_L1R = cms.Path( HLTDoublePhoton20L1NonIsolatedHLTLooseIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoublePhoton8_L1R = cms.Path( HLTDoublePhoton8L1NonIsolatedHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoublePhoton10_L1R = cms.Path( HLTDoublePhoton10L1NonIsolatedHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoublePhoton20_L1R = cms.Path( HLTDoublePhoton20L1NonIsolatedHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoublePhoton10_Exclusive = cms.Path( HLTBeginSequence + hltL1seedExclusiveDoubleEgamma + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoDoubleExclPhotonL1MatchFilterRegional + hltL1IsoDoubleExclPhotonEtPhiFilter + hltL1IsolatedPhotonEcalIsol + hltL1IsoDoubleExclPhotonEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1IsoDoubleExclPhotonHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1IsoDoubleExclPhotonTrackIsolFilter + hltDoubleExclPhotonL1IsoPresc + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1Mu = cms.Path( HLTL1muonrecoSequence + hltPrescaleMuLevel1Path + hltMuLevel1PathLevel1Seed + hltMuLevel1PathL1Filtered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1MuOpen = cms.Path( HLTL1muonrecoSequence + hltPrescaleMuLevel1Open + hltMuLevel1PathLevel1OpenSeed + hltMuLevel1PathL1OpenFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L2Mu9 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuLevel2 + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuLevel2NoIsoL2PreFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoMu9 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuIso9 + hltSingleMuIsoLevel1Seed + hltSingleMuIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuIsoL2PreFiltered7 + HLTL2muonisorecoSequence + hltSingleMuIsoL2IsoFiltered7 + HLTL3muonrecoSequence + hltSingleMuIsoL3PreFiltered9 + HLTL3muonisorecoSequence + hltSingleMuIsoL3IsoFiltered9 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoMu11 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuIso + hltSingleMuIsoLevel1Seed + hltSingleMuIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuIsoL2PreFiltered + HLTL2muonisorecoSequence + hltSingleMuIsoL2IsoFiltered + HLTL3muonrecoSequence + hltSingleMuIsoL3PreFiltered + HLTL3muonisorecoSequence + hltSingleMuIsoL3IsoFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoMu13 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuIso13 + hltSingleMuIsoLevel1Seed10 + hltSingleMuIsoL1Filtered10 + HLTL2muonrecoSequence + hltSingleMuIsoL2PreFiltered11 + HLTL2muonisorecoSequence + hltSingleMuIsoL2IsoFiltered11 + HLTL3muonrecoSequence + hltSingleMuIsoL3PreFiltered13 + HLTL3muonisorecoSequence + hltSingleMuIsoL3IsoFiltered13 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoMu15 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuIso15 + hltSingleMuIsoLevel1Seed10 + hltSingleMuIsoL1Filtered10 + HLTL2muonrecoSequence + hltSingleMuIsoL2PreFiltered12 + HLTL2muonisorecoSequence + hltSingleMuIsoL2IsoFiltered12 + HLTL3muonrecoSequence + hltSingleMuIsoL3PreFiltered15 + HLTL3muonisorecoSequence + hltSingleMuIsoL3IsoFiltered15 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_NoTrackerIsoMu15 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIso + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered + HLTL3TkmuonrecoNocandSequence + hltL3TkMuonCandidates + hltSingleMuNoIsoL3TkPreFilter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu3 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuPrescale3 + hltSingleMuPrescale3Level1Seed + hltSingleMuPrescale3L1Filtered + HLTL2muonrecoSequence + hltSingleMuPrescale3L2PreFiltered + HLTL3muonrecoSequence + hltSingleMuPrescale3L3PreFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu5 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuPrescale5 + hltSingleMuPrescale5Level1Seed + hltSingleMuPrescale5L1Filtered + HLTL2muonrecoSequence + hltSingleMuPrescale5L2PreFiltered + HLTL3muonrecoSequence + hltSingleMuPrescale5L3PreFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu7 = cms.Path( HLTL1muonrecoSequence + hltPreSingleMuPrescale77 + hltSingleMuPrescale77Level1Seed + hltSingleMuPrescale77L1Filtered + HLTL2muonrecoSequence + hltSingleMuPrescale77L2PreFiltered + HLTL3muonrecoSequence + hltSingleMuPrescale77L3PreFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu9 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIso9 + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered7 + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFiltered9 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu10 = cms.Path( HLTL1muonrecoSequence + hltPreSingleMuPrescale710 + hltSingleMuPrescale710Level1Seed + hltSingleMuPrescale710L1Filtered + HLTL2muonrecoSequence + hltSingleMuPrescale710L2PreFiltered + HLTL3muonrecoSequence + hltSingleMuPrescale710L3PreFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu11 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIso11 + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered9 + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFiltered11 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu13 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIso13 + hltSingleMuNoIsoLevel1Seed10 + hltSingleMuNoIsoL1Filtered10 + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered11 + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFiltered13 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu15 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIso15 + hltSingleMuNoIsoLevel1Seed10 + hltSingleMuNoIsoL1Filtered10 + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered12 + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFiltered15 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu15_L1Mu7 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIso + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu15_Vtx2cm = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIsoRelaxedVtx2cm + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu15_Vtx2mm = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIsoRelaxedVtx2mm + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFilteredRelaxedVtx2mm + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleIsoMu3 = cms.Path( HLTL1muonrecoSequence + hltPrescaleDiMuonIso + hltDiMuonIsoLevel1Seed + hltDiMuonIsoL1Filtered + HLTL2muonrecoSequence + hltDiMuonIsoL2PreFiltered + HLTL2muonisorecoSequence + hltDiMuonIsoL2IsoFiltered + HLTL3muonrecoSequence + hltDiMuonIsoL3PreFiltered + HLTL3muonisorecoSequence + hltDiMuonIsoL3IsoFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleMu3 = cms.Path( HLTL1muonrecoSequence + hltPrescaleDiMuonNoIso + hltDiMuonNoIsoLevel1Seed + hltDiMuonNoIsoL1Filtered + HLTL2muonrecoSequence + hltDiMuonNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltDiMuonNoIsoL3PreFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleMu3_Vtx2cm = cms.Path( HLTL1muonrecoSequence + hltPrescaleDiMuonNoIsoRelaxedVtx2cm + hltDiMuonNoIsoLevel1Seed + hltDiMuonNoIsoL1Filtered + HLTL2muonrecoSequence + hltDiMuonNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltDiMuonNoIsoL3PreFilteredRelaxedVtx2cm + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleMu3_Vtx2mm = cms.Path( HLTL1muonrecoSequence + hltPrescaleDiMuonNoIsoRelaxedVtx2mm + hltDiMuonNoIsoLevel1Seed + hltDiMuonNoIsoL1Filtered + HLTL2muonrecoSequence + hltDiMuonNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleMu3_JPsi = cms.Path( HLTL1muonrecoSequence + hltPrescaleJPsiMM + hltJpsiMMLevel1Seed + hltJpsiMML1Filtered + HLTL2muonrecoSequence + hltJpsiMML2Filtered + HLTL3muonrecoSequence + hltJpsiMML3Filtered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleMu3_Upsilon = cms.Path( HLTL1muonrecoSequence + hltPrescaleUpsilonMM + hltUpsilonMMLevel1Seed + hltUpsilonMML1Filtered + HLTL2muonrecoSequence + hltUpsilonMML2Filtered + HLTL3muonrecoSequence + hltUpsilonMML3Filtered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleMu7_Z = cms.Path( HLTL1muonrecoSequence + hltPrescaleZMM + hltZMMLevel1Seed + hltZMML1Filtered + HLTL2muonrecoSequence + hltZMML2Filtered + HLTL3muonrecoSequence + hltZMML3Filtered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleMu3_SameSign = cms.Path( HLTL1muonrecoSequence + hltPrescaleSameSignMu + hltSameSignMuLevel1Seed + hltSameSignMuL1Filtered + HLTL2muonrecoSequence + hltSameSignMuL2PreFiltered + HLTL3muonrecoSequence + hltSameSignMuL3PreFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleMu3_Psi2S = cms.Path( HLTBeginSequence + hltPrescalePsi2SMM + hltJpsiMMLevel1Seed + hltJpsiMML1Filtered + HLTL2muonrecoSequence + hltPsi2SMML2Filtered + HLTL3muonrecoSequence + hltPsi2SMML3Filtered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_TripleMu3 = cms.Path( HLTL1muonrecoSequence + hltPrescaleMultiMuonNoIso + hltMultiMuonNoIsoLevel1Seed + hltMultiMuonNoIsoL1Filtered + HLTL2muonrecoSequence + hltMultiMuonNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltMultiMuonNoIsoL3PreFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_Jet180 = cms.Path( HLTBeginSequence + hltPrescalerBLifetime1jet + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime1jetL2filter + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_Jet120_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBLifetime1jet120 + hltBLifetimeL1seedsLowEnergy + HLTBCommonL2recoSequence + hltBLifetime1jetL2filter120 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_Jet160_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBLifetime1jet160 + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime1jetL2filter160 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_DoubleJet120 = cms.Path( HLTBeginSequence + hltPrescalerBLifetime2jet + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime2jetL2filter + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_DoubleJet60_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBLifetime2jet60 + hltBLifetimeL1seedsLowEnergy + HLTBCommonL2recoSequence + hltBLifetime2jetL2filter60 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_DoubleJet100_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBLifetime2jet100 + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime2jetL2filter100 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_TripleJet70 = cms.Path( HLTBeginSequence + hltPrescalerBLifetime3jet + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime3jetL2filter + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_TripleJet60_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBLifetime3jet60 + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime3jetL2filter60 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_TripleJet40_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBLifetime3jet40 + hltBLifetimeL1seedsLowEnergy + HLTBCommonL2recoSequence + hltBLifetime3jetL2filter40 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_QuadJet40 = cms.Path( HLTBeginSequence + hltPrescalerBLifetime4jet + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime4jetL2filter + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_QuadJet35_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBLifetime4jet35 + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime4jetL2filter35 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_QuadJet30_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBLifetime4jet30 + hltBLifetimeL1seedsLowEnergy + HLTBCommonL2recoSequence + hltBLifetime4jetL2filter30 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_HT470 = cms.Path( HLTBeginSequence + hltPrescalerBLifetimeHT + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetimeHTL2filter + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_HT420_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBLifetimeHT420 + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetimeHTL2filter420 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_HT320_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBLifetimeHT320 + hltBLifetimeL1seedsLowEnergy + HLTBCommonL2recoSequence + hltBLifetimeHTL2filter320 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_DoubleJet120 = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon2jet + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon2jetL2filter + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_DoubleJet100_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon2jet100 + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon2jetL2filter100 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_DoubleJet60_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon2jet60 + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon2jetL2filter60 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_TripleJet70 = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon3jet + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon3jetL2filter + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_TripleJet60_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon3jet60 + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon3jetL2filter60 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_TripleJet40_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon3jet40 + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon3jetL2filter40 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_QuadJet40 = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon4jet + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon4jetL2filter + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_QuadJet35_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon4jet35 + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon4jetL2filter35 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_QuadJet30_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon4jet30 + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon4jetL2filter30 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_HT370 = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuonHT + hltBSoftmuonHTL1seeds + HLTBCommonL2recoSequence + hltBSoftmuonHTL2filter + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_HT330_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuonHT330 + hltBSoftmuonHTL1seeds + HLTBCommonL2recoSequence + hltBSoftmuonHTL2filter330 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_HT250_Relaxed = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuonHT250 + hltBSoftmuonHTL1seedsLowEnergy + HLTBCommonL2recoSequence + hltBSoftmuonHTL2filter250 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleMu3_BJPsi = cms.Path( HLTBeginSequence + hltJpsitoMumuL1SeedRelaxed + hltJpsitoMumuL1FilteredRelaxed + HLTL2muonrecoSequence + HLTL3displacedMumurecoSequence + hltDisplacedJpsitoMumuFilterRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleMu4_BJPsi = cms.Path( HLTBeginSequence + hltJpsitoMumuL1Seed + hltJpsitoMumuL1Filtered + HLTL2muonrecoSequence + HLTL3displacedMumurecoSequence + hltDisplacedJpsitoMumuFilter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_TripleMu3_TauTo3Mu = cms.Path( HLTBeginSequence + hltMuMukL1Seed + hltMuMukL1Filtered + HLTL2muonrecoSequence + HLTL3displacedMumurecoSequence + hltDisplacedMuMukFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTRecopixelvertexingSequence + hltMumukPixelSeedFromL2Candidate + hltCkfTrackCandidatesMumuk + hltCtfWithMaterialTracksMumuk + hltMumukAllConeTracks + hltmmkFilter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoTau_MET65_Trk20 = cms.Path( HLTBeginSequence + hltSingleTauPrescaler + hltSingleTauL1SeedFilter + HLTCaloTausCreatorSequence + hltMet + hlt1METSingleTau + hltL2SingleTauJets + hltL2SingleTauIsolationProducer + hltL2SingleTauIsolationSelector + hltFilterSingleTauEcalIsolation + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltAssociatorL25SingleTau + hltConeIsolationL25SingleTau + hltIsolatedL25SingleTau + hltFilterL25SingleTau + HLTDoLocalStripSequence + hltL3SingleTauPixelSeeds + hltCkfTrackCandidatesL3SingleTau + hltCtfWithMaterialTracksL3SingleTau + hltAssociatorL3SingleTau + hltConeIsolationL3SingleTau + hltIsolatedL3SingleTau + hltFilterL3SingleTau + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoTau_MET35_Trk15_L1MET = cms.Path( HLTBeginSequence + hltSingleTauMETPrescaler + hltSingleTauMETL1SeedFilter + HLTCaloTausCreatorSequence + hltMet + hlt1METSingleTauMET + hltL2SingleTauMETJets + hltL2SingleTauMETIsolationProducer + hltL2SingleTauMETIsolationSelector + hltFilterSingleTauMETEcalIsolation + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltAssociatorL25SingleTauMET + hltConeIsolationL25SingleTauMET + hltIsolatedL25SingleTauMET + hltFilterL25SingleTauMET + HLTDoLocalStripSequence + hltL3SingleTauMETPixelSeeds + hltCkfTrackCandidatesL3SingleTauMET + hltCtfWithMaterialTracksL3SingleTauMET + hltAssociatorL3SingleTauMET + hltConeIsolationL3SingleTauMET + hltIsolatedL3SingleTauMET + hltFilterL3SingleTauMET + cms.SequencePlaceholder("HLTEndSequence") )
HLT_LooseIsoTau_MET30 = cms.Path( HLTBeginSequence + hltSingleTauPrescaler + hltSingleTauL1SeedFilter + HLTCaloTausCreatorSequence + hltMet + hlt1METSingleTauRelaxed + hltL2SingleTauJets + hltL2SingleTauIsolationProducer + hltL2SingleTauIsolationSelectorRelaxed + hltFilterSingleTauEcalIsolationRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_LooseIsoTau_MET30_L1MET = cms.Path( HLTBeginSequence + hltSingleTauMETPrescaler + hltSingleTauMETL1SeedFilter + HLTCaloTausCreatorSequence + hltMet + hlt1METSingleTauMETRelaxed + hltL2SingleTauMETJets + hltL2SingleTauMETIsolationProducer + hltL2SingleTauMETIsolationSelectorRelaxed + hltFilterSingleTauMETEcalIsolationRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleIsoTau_Trk3 = cms.Path( HLTBeginSequence + hltDoubleTauPrescaler + hltDoubleTauL1SeedFilter + HLTCaloTausCreatorRegionalSequence + hltL2DoubleTauJets + hltL2DoubleTauIsolationProducer + hltL2DoubleTauIsolationSelector + hltFilterDoubleTauEcalIsolation + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltAssociatorL25PixelTauIsolated + hltConeIsolationL25PixelTauIsolated + hltIsolatedL25PixelTauPtLeadTk + hltFilterL25PixelTauPtLeadTk + hltIsolatedL25PixelTau + hltFilterL25PixelTau + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleLooseIsoTau = cms.Path( HLTBeginSequence + hltDoubleTauPrescaler + hltDoubleTauL1SeedFilterRelaxed + HLTCaloTausCreatorRegionalSequence + hltL2DoubleTauJetsRelaxed + hltL2DoubleTauIsolationProducerRelaxed + hltL2DoubleTauIsolationSelectorRelaxed + hltFilterDoubleTauEcalIsolationRelaxed + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle8_IsoMu7 = cms.Path( HLTBeginSequence + hltemuPrescale + hltEMuonLevel1Seed + hltEMuL1MuonFilter + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltemuL1IsoSingleL1MatchFilter + hltemuL1IsoSingleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltemuL1IsoSingleElectronHcalIsolFilter + HLTL2muonrecoSequence + hltEMuL2MuonPreFilter + HLTL2muonisorecoSequence + hltEMuL2MuonIsoFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltemuL1IsoSingleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltemuL1IsoSingleElectronEoverpFilter + HLTL3muonrecoSequence + hltEMuL3MuonPreFilter + HLTL3muonisorecoSequence + hltEMuL3MuonIsoFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltemuL1IsoSingleElectronTrackIsolFilter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle10_Mu10_L1R = cms.Path( HLTBeginSequence + hltemuNonIsoPrescale + hltemuNonIsoLevel1Seed + hltNonIsoEMuL1MuonFilter + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltemuNonIsoL1MatchFilterRegional + hltemuNonIsoL1IsoEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltemuNonIsoL1HcalIsolFilter + HLTL2muonrecoSequence + hltNonIsoEMuL2MuonPreFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + HLTPixelMatchElectronL1NonIsoSequence + hltemuNonIsoL1IsoPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoTrackingSequence + hltemuNonIsoL1IsoEoverpFilter + HLTL3muonrecoSequence + hltNonIsoEMuL3MuonPreFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + HLTL1NonIsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1NonIsoElectronTrackIsol + hltemuNonIsoL1IsoTrackIsolFilter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle12_IsoTau_Trk3 = cms.Path( HLTBeginSequence + hltPrescalerElectronTau + hltLevel1GTSeedElectronTau + HLTETauSingleElectronL1IsolatedHOneOEMinusOneOPFilterSequence + HLTL2TauJetsElectronTauSequnce + hltL2ElectronTauIsolationProducer + hltL2ElectronTauIsolationSelector + hltFilterEcalIsolatedTauJetsElectronTau + HLTRecopixelvertexingSequence + hltJetTracksAssociatorAtVertexL25ElectronTau + hltConeIsolationL25ElectronTau + hltIsolatedTauJetsSelectorL25ElectronTauPtLeadTk + hltFilterIsolatedTauJetsL25ElectronTauPtLeadTk  + hltIsolatedTauJetsSelectorL25ElectronTau + hltFilterIsolatedTauJetsL25ElectronTau + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle10_BTagIP_Jet35 = cms.Path( HLTBeginSequence + hltElectronBPrescale + hltElectronBL1Seed + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltElBElectronL1MatchFilter + hltElBElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltElBElectronHcalIsolFilter + HLTBCommonL2recoSequence + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1NonIsoSequence + HLTPixelMatchElectronL1IsoSequence + hltElBElectronPixelMatchFilter + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + HLTPixelMatchElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoTrackingSequence + hltElBElectronEoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + HLTL1NonIsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1NonIsoElectronTrackIsol + hltElBElectronTrackIsolFilter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle12_Jet40 = cms.Path( HLTL1EplusJetSequence + HLTEJetElectronSequence + HLTDoCaloSequence + HLTDoJetRecoSequence + hltej1jet40 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle12_DoubleJet80 = cms.Path( HLTL1EplusJetSequence + HLTEJetElectronSequence + HLTDoCaloSequence + HLTDoJetRecoSequence + hltej2jet80 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoElec5_TripleJet30 = cms.Path( HLTL1EplusJet30Sequence + HLTE3Jet30ElectronSequence + HLTDoCaloSequence + HLTDoJetRecoSequence + hltej3jet30 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle12_TripleJet60 = cms.Path( HLTL1EplusJetSequence + HLTEJetElectronSequence + HLTDoCaloSequence + HLTDoJetRecoSequence + hltej3jet60 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoEle12_QuadJet35 = cms.Path( HLTL1EplusJetSequence + HLTEJetElectronSequence + HLTDoCaloSequence + HLTDoJetRecoSequence + hltej4jet35 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoMu14_IsoTau_Trk3 = cms.Path( HLTBeginSequence + hltPrescalerMuonTau + hltLevel1GTSeedMuonTau + hltMuonTauL1Filtered + HLTL2muonrecoSequence + hltMuonTauIsoL2PreFiltered + HLTL2muonisorecoSequence + hltMuonTauIsoL2IsoFiltered + HLTDoLocalStripSequence + HLTL3muonrecoSequence + HLTL3muonisorecoSequence + hltMuonTauIsoL3PreFiltered + hltMuonTauIsoL3IsoFiltered + HLTCaloTausCreatorRegionalSequence + hltL2TauJetsProviderMuonTau + hltL2MuonTauIsolationProducer + hltL2MuonTauIsolationSelector + hltFilterEcalIsolatedTauJetsMuonTau + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltJetsPixelTracksAssociatorMuonTau + hltPixelTrackConeIsolationMuonTau + hltIsolatedTauJetsSelectorL25MuonTauPtLeadTk + hltFilterL25MuonTauPtLeadTk + hltPixelTrackIsolatedTauJetsSelectorMuonTau + hltFilterPixelTrackIsolatedTauJetsMuonTau + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoMu7_BTagIP_Jet35 = cms.Path( HLTBeginSequence + hltMuBPrescale + hltMuBLevel1Seed + hltMuBLifetimeL1Filtered + HLTL2muonrecoSequence + hltMuBLifetimeIsoL2PreFiltered + HLTL2muonisorecoSequence + hltMuBLifetimeIsoL2IsoFiltered + HLTBCommonL2recoSequence + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTL3muonrecoSequence + hltMuBLifetimeIsoL3PreFiltered + HLTL3muonisorecoSequence + hltMuBLifetimeIsoL3IsoFiltered + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoMu7_BTagMu_Jet20 = cms.Path( HLTBeginSequence + hltMuBsoftMuPrescale + hltMuBLevel1Seed + hltMuBSoftL1Filtered + HLTL2muonrecoSequence + hltMuBSoftIsoL2PreFiltered + HLTL2muonisorecoSequence + hltMuBSoftIsoL2IsoFiltered + HLTBCommonL2recoSequence + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTL3muonrecoSequence + hltMuBSoftIsoL3PreFiltered + HLTL3muonisorecoSequence + hltMuBSoftIsoL3IsoFiltered + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoMu7_Jet40 = cms.Path( HLTL1muonrecoSequence + hltMuJetsPrescale + hltMuJetsLevel1Seed + hltMuJetsL1Filtered + HLTL2muonrecoSequence + hltMuJetsL2PreFiltered + HLTL2muonisorecoSequence + hltMuJetsL2IsoFiltered + HLTL3muonrecoSequence + hltMuJetsL3PreFiltered + HLTL3muonisorecoSequence + hltMuJetsL3IsoFiltered + HLTDoCaloSequence + HLTDoJetRecoSequence + hltMuJetsHLT1jet40 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_NoL2IsoMu8_Jet40 = cms.Path( HLTL1muonrecoSequence + hltMuNoL2IsoJetsPrescale + hltMuNoL2IsoJetsLevel1Seed + hltMuNoL2IsoJetsL1Filtered + HLTL2muonrecoSequence + hltMuNoL2IsoJetsL2PreFiltered + HLTL3muonrecoSequence + hltMuNoL2IsoJetsL3PreFiltered + HLTL3muonisorecoSequence + hltMuNoL2IsoJetsL3IsoFiltered + HLTDoCaloSequence + HLTDoJetRecoSequence + hltMuNoL2IsoJetsHLT1jet40 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu14_Jet50 = cms.Path( HLTL1muonrecoSequence + hltMuNoIsoJetsPrescale + hltMuNoIsoJetsLevel1Seed + hltMuNoIsoJetsL1Filtered + HLTL2muonrecoSequence + hltMuNoIsoJetsL2PreFiltered + HLTL3muonrecoSequence + hltMuNoIsoJetsL3PreFiltered + HLTDoCaloSequence + HLTDoJetRecoSequence + hltMuNoIsoJetsHLT1jet50 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu5_TripleJet30 = cms.Path( HLTL1muonrecoSequence + hltMuNoIsoJetsPrescale + hltMuNoIsoJets30Level1Seed + hltMuNoIsoJetsMinPt4L1Filtered + HLTL2muonrecoSequence + hltMuNoIsoJetsMinPt4L2PreFiltered + HLTL3muonrecoSequence + hltMuNoIsoJetsMinPtL3PreFiltered + HLTDoCaloSequence + HLTDoJetRecoSequence + hltMuNoIsoHLTJets3jet30 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_Jet20_Calib = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon1jet + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon1jetL2filter + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonByDRL3filter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_ZeroBias = cms.Path( HLTBeginSequence + hltl1sZero + hltpreZero + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MinBias = cms.Path( HLTBeginSequence + hltl1sMin + hltpreMin + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MinBiasHcal = cms.Path( HLTBeginSequence + hltl1sMinHcal + hltpreMinHcal + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MinBiasEcal = cms.Path( HLTBeginSequence + hltl1sMinEcal + hltpreMinEcal + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MinBiasPixel = cms.Path( HLTBeginSequence + hltPreMinBiasPixel + hltL1seedMinBiasPixel + HLTDoLocalPixelSequence + HLTPixelTrackingForMinBiasSequence + hltPixelCands + hltMinBiasPixelFilter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MinBiasPixel_Trk5 = cms.Path( HLTBeginSequence + hltPreMBForAlignment + hltL1seedMinBiasPixel + HLTDoLocalPixelSequence + HLTPixelTrackingForMinBiasSequence + hltPixelCands + hltPixelMBForAlignment + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BackwardBSC = cms.Path( HLTBeginSequence + hltLevel1seedHLTBackwardBSC + hltPrescaleHLTBackwardBSC + cms.SequencePlaceholder("HLTEndSequence") )
HLT_ForwardBSC = cms.Path( HLTBeginSequence + hltLevel1seedHLTForwardBSC + hltPrescaleHLTForwardBSC + cms.SequencePlaceholder("HLTEndSequence") )
HLT_CSCBeamHalo = cms.Path( HLTBeginSequence + hltLevel1seedHLTCSCBeamHalo + hltPrescaleHLTCSCBeamHalo + cms.SequencePlaceholder("HLTEndSequence") )
HLT_CSCBeamHaloOverlapRing1 = cms.Path( HLTBeginSequence + hltLevel1seedHLTCSCBeamHaloOverlapRing1 + hltPrescaleHLTCSCBeamHaloOverlapRing1 + cms.SequencePlaceholder("simMuonCSCDigis") + hltCsc2DRecHits + hltOverlapsHLTCSCBeamHaloOverlapRing1 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_CSCBeamHaloOverlapRing2 = cms.Path( HLTBeginSequence + hltLevel1seedHLTCSCBeamHaloOverlapRing2 + hltPrescaleHLTCSCBeamHaloOverlapRing2 + cms.SequencePlaceholder("simMuonCSCDigis") + hltCsc2DRecHits + hltOverlapsHLTCSCBeamHaloOverlapRing2 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_CSCBeamHaloRing2or3 = cms.Path( HLTBeginSequence + hltLevel1seedHLTCSCBeamHaloRing2or3 + hltPrescaleHLTCSCBeamHaloRing2or3 + cms.SequencePlaceholder("simMuonCSCDigis") + hltCsc2DRecHits + hltFilter23HLTCSCBeamHaloRing2or3 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_TrackerCosmics = cms.Path( HLTBeginSequence + hltLevel1seedHLTTrackerCosmics + hltPrescaleHLTTrackerCosmics + cms.SequencePlaceholder("HLTEndSequence") )
HLT_TriggerType = cms.Path( HLTBeginSequence + hltPrescaleTriggerType + hltFilterTriggerType + cms.SequencePlaceholder("HLTEndSequence") )
AlCa_IsoTrack = cms.Path( HLTBeginSequence + HLTL1SeedFilterSequence + hltPreIsolTrackNoEcalIso + HLTL3PixelIsolFilterSequence + HLTIsoTrRegFEDSelection + cms.SequencePlaceholder("HLTEndSequence") )
AlCa_EcalPhiSym = cms.Path( HLTBeginSequence + hltL1sEcalPhiSym + hltEcalPhiSymPresc + hltEcalDigis + hltEcalWeightUncalibRecHit + hltEcalRecHit + hltAlCaPhiSymStream + cms.SequencePlaceholder("HLTEndSequence") )
AlCa_EcalPi0 = cms.Path( HLTBeginSequence + hltPrePi0Ecal + hltL1sEcalPi0 + HLTDoRegionalEgammaEcalSequence + hltAlCaPi0RegRecHits + cms.SequencePlaceholder("HLTEndSequence") )
HLTriggerFinalPath = cms.Path( hltTriggerSummaryAOD + hltTriggerSummaryRAWprescaler + hltTriggerSummaryRAW + hltBoolFinal )
HLTEndpath1 = cms.EndPath( hltL1gtTrigReport + hltTrigReport )




