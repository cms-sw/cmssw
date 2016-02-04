#!/usr/bin/env python

# Usage: ./getFastSimHLTcff.py <Version from ConfDB> <Name of cff file> <optional L1 Menu> <optional subset of paths>
###TABLE FOR 8E29

import sys
import os
import commands
import getopt
import fileinput

def usage():
    print "Usage: ./getFastSimHLT_GRun_cff.py <Version from ConfDB> <Name of cff> <optional L1 Menu> <optional subset of paths>"
    print "       Default L1 Menu: L1Menu_Commissioning2009_v0"
    print "       Define subset of paths as comma-separated list: a,b,c (Default is to run all paths)"
    sys.exit(1)

argc = len(sys.argv)
blockName = "None"
#cfgName = "None"
usePaths = "All"
L1Menu   = "L1Menu_Commissioning2009_v0"

if argc == 3:
    dbName  = sys.argv[1]
    cffName = sys.argv[2]
elif argc == 4:
    dbName  = sys.argv[1]
    cffName = sys.argv[2]
    blockName = sys.argv[3]
elif argc == 5:
    dbName  = sys.argv[1]
    cffName = sys.argv[2]
    blockName = sys.argv[3]
    L1Menu  = sys.argv[4]
elif argc == 6:
    dbName  = sys.argv[1]
    cffName = sys.argv[2]
    blockName = sys.argv[3]
    L1Menu   = sys.argv[4]
    usePaths = sys.argv[5]
else:
    usage()

if os.path.exists(cffName):
    print cffName, "already exists.  WARNING: No new file created!"
else:
    essources = "--essources "
    essources += "-es_hardcode,"
    essources += "-magfield,"
    essources += "-eegeom,"
    essources += "-XMLIdealGeometryESSource,"
    essources += "-HepPDTESSource,"
    essources += "-BTagRecord,"
    essources += "-GlobalTag,"
    essources += "-Level1MenuOverride"

    esmodules = "--esmodules "
    # strip automatic magnetic field definition
    esmodules += "-AutoMagneticFieldESProducer,"
    esmodules += "-SlaveField0,"
    esmodules += "-SlaveField20,"
    esmodules += "-SlaveField30,"
    esmodules += "-SlaveField35,"
    esmodules += "-SlaveField38,"
    esmodules += "-SlaveField40,"
    esmodules += "-VBF0,"
    esmodules += "-VBF20,"
    esmodules += "-VBF30,"
    esmodules += "-VBF35,"
    esmodules += "-VBF38,"
    esmodules += "-VBF40,"
    # strip static magnetic field definition
    esmodules += "-VolumeBasedMagneticFieldESProducer,"
    esmodules += "-ParametrizedMagneticFieldProducer,"
    esmodules += "-SiPixelTemplateDBObjectESProducer,"
    esmodules += "-TTRHBuilderPixelOnly,"
    esmodules += "-WithTrackAngle,"
    esmodules += "-trajectoryCleanerBySharedHits,"
    esmodules += "-trackCounting3D2nd,"
    esmodules += "-navigationSchoolESProducer,"
    esmodules += "-muonCkfTrajectoryFilter,"
    esmodules += "-l1GtTriggerMenuXml,"
    esmodules += "-L1GtTriggerMaskAlgoTrigTrivialProducer,"
    esmodules += "-L1GtTriggerMaskTechTrigTrivialProducer,"
    esmodules += "-hcal_db_producer,"
    esmodules += "-ckfBaseTrajectoryFilter,"
    esmodules += "-ZdcHardcodeGeometryEP,"
    esmodules += "-TransientTrackBuilderESProducer,"
    esmodules += "-TrackerRecoGeometryESProducer,"
    esmodules += "-TrackerGeometricDetESModule,"
    esmodules += "-TrackerDigiGeometryESModule,"
    esmodules += "-StripCPEfromTrackAngleESProducer,"
    esmodules += "-SteppingHelixPropagatorOpposite,"
    esmodules += "-SteppingHelixPropagatorAny,"
    esmodules += "-SteppingHelixPropagatorAlong,"
    esmodules += "-SmootherRK,"
    esmodules += "-SmartPropagatorRK,"
    esmodules += "-SmartPropagatorOpposite,"
    esmodules += "-SmartPropagatorAnyRK,"
    esmodules += "-SmartPropagatorAnyOpposite,"
    esmodules += "-SmartPropagatorAny,"
    esmodules += "-SmartPropagator,"
    esmodules += "-SiStripRecHitMatcherESProducer,"
    esmodules += "-SiStripGainESProducer,"
    esmodules += "-SiStripQualityESProducer,"
    esmodules += "-RungeKuttaTrackerPropagator,"
    esmodules += "-RPCGeometryESModule,"
    esmodules += "-OppositeMaterialPropagator,"
    esmodules += "-MuonTransientTrackingRecHitBuilderESProducer,"
    esmodules += "-MuonNumberingInitialization,"
    esmodules += "-MuonDetLayerGeometryESProducer,"
    esmodules += "-MuonCkfTrajectoryBuilder,"
    esmodules += "-hltMeasurementTracker,"
    esmodules += "-MaterialPropagator,"
    esmodules += "-L3MuKFFitter,"
    esmodules += "-KFUpdatorESProducer,"
    esmodules += "-KFSmootherForRefitInsideOut,"
    esmodules += "-KFSmootherForMuonTrackLoader,"
    esmodules += "-KFFitterForRefitInsideOut,"
    esmodules += "-HcalTopologyIdealEP,"
    esmodules += "-HcalHardcodeGeometryEP,"
    esmodules += "-GroupedCkfTrajectoryBuilder,"
    esmodules += "-GlobalTrackingGeometryESProducer,"
    esmodules += "-FittingSmootherRK,"
    esmodules += "-FitterRK,"
    esmodules += "-EcalPreshowerGeometryEP,"
    esmodules += "-EcalLaserCorrectionService,"
    esmodules += "-EcalEndcapGeometryEP,"
    esmodules += "-EcalElectronicsMappingBuilder,"
    esmodules += "-EcalBarrelGeometryEP,"
    esmodules += "-DTGeometryESModule,"
    esmodules += "-hltCkfTrajectoryBuilder,"
    esmodules += "-Chi2MeasurementEstimator,"
    esmodules += "-Chi2EstimatorForRefit,"
    esmodules += "-CaloTowerHardcodeGeometryEP,"
    esmodules += "-CaloTowerConstituentsMapBuilder,"
    esmodules += "-CaloTopologyBuilder,"
    esmodules += "-CaloGeometryBuilder,"
    esmodules += "-CSCGeometryESModule,"
# Added since re-labelling of many ES plugins used by HLT in  V11-02-00  HLTrigger/Configuration
    esmodules += "-hltESPEcalTrigTowerConstituentsMapBuilder,"
    esmodules += "-hltESPMuonDetLayerGeometryESProducer,"
    esmodules += "-hltESPTrackerRecoGeometryESProducer,"
    esmodules += "-hltESPGlobalTrackingGeometryESProducer"

    #--- Define blocks for ElectronPixelSeeds ---#
    #--- Please see FastSimulation/EgammaElectronAlgos/data/pixelMatchElectronL1[Non]Iso[LargeWindow]SequenceForHLT.cff ---#
    blocks = "--blocks "
    blocks += "hltL1NonIsoLargeWindowElectronPixelSeeds::SeedConfiguration,"
    blocks += "hltL1IsoLargeWindowElectronPixelSeeds::SeedConfiguration,"
    blocks += "hltL1NonIsoStartUpElectronPixelSeeds::SeedConfiguration,"
    blocks += "hltL1IsoStartUpElectronPixelSeeds::SeedConfiguration,"


    #--- Some notes about removed/redefined modules ---#
    #--- hltGtDigis --> FastSim uses gtDigis from L1Trigger/Configuration/data/L1Emulator.cff
    #--- hltMuon[CSC/DT/RPC]Digis defined (as muon[CSC/DT/RPC]Digis) through FastSimulation/Configuration/data/FamosSequences.cff
    #--- hltL1[Non]IsoElectronPixelSeeds -> defined in FastSimulation/EgammaElectronAlgos/data/pixelMatchElectronL1[Non]IsoSequenceForHLT.cff
    #--- hltCtfL1[Non]IsoWithMaterialTracks, hltCkfL1[Non]IsoTrackCandidates -> defined in FastSimulation/EgammaElectronAlgos/data/pixelMatchElectronL1[Non]IsoSequenceForHLT.cff
    #--- hltL1[Non]IsoLargeWindowElectronPixelSeeds -> defined in FastSimulation/EgammaElectronAlgos/data/pixelMatchElectronL1[Non]IsoLargeWindowSequenceForHLT.cff
    #--- hltCtfL1[Non]IsoLargeWindowWithMaterialTracks, hltCkfL1[Non]IsoLargeWindowTrackCandidates -> defined in FastSimulation/EgammaElectronAlgos/data/pixelMatchElectronL1[Non]IsoLargeWindowSequenceForHLT.cff
    #--- hltEcalPreshowerDigis,hltEcalRegionalJetsFEDs,hltEcalRegionalJetsDigis,hltEcalRegionalJetsWeightUncalibRecHit,hltEcalRegionalJetsRecHitTmp redefined as dummyModules in FastSimulation/HighLevelTrigger/data/common/HLTSetup.cff
    #--- hltEcalRegional[Muons/Rest]FEDs,hltEcalRegional[Muons/Rest]Digis,hltEcalRegional[Muons/Rest]WeightUncalibRecHit,hltEcalRegional[Muons/Rest]RecHitTmp redefined as dummyModules in FastSimulation/HighLevelTrigger/data/common/HLTSetup.cff
    #--- hltL3Muons defined in FastSimulation/HighLevelTrigger/data/Muon/HLTFastRecoForMuon.cff
    #--- hltL3TrajectorySeed redefined as dummyModule in FastSimulation/HighLevelTrigger/data/common/HLTSetup.cff
    #--- hltL3TrackCandidateFromL2 redefined as a sequence in FastSimulation/HighLevelTrigger/data/Muon/HLTFastRecoForMuon.cff
    #--- hltHcalDigis redefined as dummyModule in FastSimulation/HighLevelTrigger/data/common/HLTSetup.cff
    #--- hlt[Hbhe/Ho/Hf]reco --> defined in FastSimulation/HighLevelTrigger/data/common/RecoLocalCalo.cff
    #--- hltEcal[Preshower]RecHit --> defined in FastSimulation/HighLevelTrigger/data/common/RecoLocalCalo.cff
    #--- hltEcalRegional[Jets/Egamma/Muons/Taus]RecHit --> defined in FastSimulation/HighLevelTrigger/data/common/EcalRegionalReco.cff
    #--- hlt[Ecal/ES]RecHitAll --> defined in FastSimulation/HighLevelTrigger/data/common/EcalRegionalReco.cff
    #--- hltPixelVertices --> kept when HLTRecopixelvertexingSequence removed from HLT.cff
    #--- hltCtfWithMaterialTracksMumu,hltMuTracks defined in FastSimulation/HighLevelTrigger/data/btau/L3ForDisplacedMumuTrigger.cff
    #--- hltEcalDigis, hltEcalWeightUncalibRecHit redefined as dummyModules in FastSimulation/HighLevelTrigger/data/common/HLTSetup.cff
    #--- hltL3SingleTau[MET]PixelSeeds[Relaxed] redefined as dummyModules in FastSimulation/HighLevelTrigger/data/common/HLTSetup.cff
    #--- hltCkfTrackCandidatesL3SingleTau[MET][Relaxed] replaced by sequences in FastSimulation/HighLevelTrigger/data/btau/HLTFastRecoForTau.cff
    #--- hltCtfWithMaterialTracksL3SingleTau[MET][Relaxed] now defined in FastSimulation/HighLevelTrigger/data/btau/HLTFastRecoForTau.cff
    #--- hltBLifetimeRegionalPixelSeedGenerator[Relaxed] redefined as dummyModules in FastSimulation/HighLevelTrigger/data/common/HLTSetup.cff
    #--- hltBLifetimeRegionalCkfTrackCandidates[Relaxed] replaced by sequences in FastSimulation/HighLevelTrigger/data/btau/lifetimeRegionalTracking.cff
    #--- hltBLifetimeRegionalCtfWithMaterialTracks[Relaxed] redefined in FastSimulation/HighLevelTrigger/data/btau/lifetimeRegionalTracking.cff
    #--- hltMumuPixelSeedFromL2Candidate redefined as dummyModule in FastSimulation/HighLevelTrigger/data/common/HLTSetup.cff
    #--- hltCkfTrackCandidatesMumu replaced by sequence in FastSimulation/HighLevelTrigger/data/btau/L3ForDisplacedMumuTrigger.cff
    #--- hltCtfWithMaterialTracksMumu redefined in FastSimulation/HighLevelTrigger/data/btau/L3ForDisplacedMumuTrigger.cff
    #--- hltMumukPixelSeedFromL2Candidate redefined as dummyModule in FastSimulation/HighLevelTrigger/data/common/HLTSetup.cff
    #--- hltCkfTrackCandidatesMumuk replaced by sequence in FastSimulation/HighLevelTrigger/data/btau/L3ForMuMuk.cff
    #--- hltCtfWithMaterialTracksMumuk,hltMumukAllConeTracks defined in FastSimulation/HighLevelTrigger/data/btau/L3ForMuMuk.cff
    #--- hltL3MuonIsolations -> Kept when HLTL3muonisorecoSequence removed from HLT.cff
    #--- hltPixelTracksForMinBias -> defined in FastSimulation/HighLevelTrigger/data/special/HLTFastRecoForSpecial.cff
    #--- hltFilterTriggerType defined in FastSimulation/HighLevelTrigger/data/special/HLTFastRecoForSpecial.cff
    #--- hltL1[Non]IsoStartUpElectronPixelSeeds -> defined in FastSimulation/EgammaElectronAlgosFastSimulation/data/pixelMatchElectronL1[Non]IsoSequenceForHLT.cff
    #--- hltCkfL1[Non]IsoStartUpTrackCandidates, hltCtfL1[Non]IsoStartUpWithMaterialTracks -> defined in FastSimulation/EgammaElectronAlgos/data/pixelMatchElectronL1[Non]IsoSequenceForHLT.cff
    modules = "--modules "
    modules += "hltL3MuonIsolations,"
    modules += "hltPixelVertices,"
    modules += "-hltCkfL1IsoTrackCandidates,"
    modules += "-hltCtfL1IsoWithMaterialTracks,"
    modules += "-hltCkfL1NonIsoTrackCandidates,"
    modules += "-hltCtfL1NonIsoWithMaterialTracks,"
    modules += "hltPixelMatchLargeWindowElectronsL1Iso,"
    modules += "hltPixelMatchLargeWindowElectronsL1NonIso,"
    modules += "-hltESRegionalEgammaRecHit,"
    modules += "-hltEcalRegionalJetsFEDs,"
    modules += "-hltEcalRegionalJetsRecHitTmp,"
    modules += "-hltEcalRegionalMuonsFEDs,"
    modules += "-hltEcalRegionalMuonsRecHitTmp,"
    modules += "-hltEcalRegionalEgammaFEDs,"
    modules += "-hltEcalRegionalEgammaRecHitTmp,"
    modules += "-hltFEDSelector,"
    #--- ADAM for muon cascade algo
    #modules += "-hltL3Muons,"
    #modules += "-hltL3TrajectorySeed,"
    #modules += "-hltL3TrackCandidateFromL2,"
    #modules += "-hltL3TkTracksFromL2,"
    modules += "-hltL3TrajSeedOIHit,"
    modules += "-hltL3TrajSeedIOHit,"
    modules += "-hltL3TrackCandidateFromL2OIState,"
    modules += "-hltL3TrackCandidateFromL2OIHit,"
    modules += "-hltL3TrackCandidateFromL2IOHit,"
    modules += "-hltL3TrackCandidateFromL2NoVtx,"
    modules += "-hltHcalDigis,"
    modules += "-hltHoreco,"
    modules += "-hltHfreco,"
    modules += "-hltHbhereco,"
    modules += "-hltEcalRegionalRestFEDs,"
    modules += "-hltEcalRegionalESRestFEDs,"
    modules += "-hltEcalRawToRecHitFacility,"
    modules += "-hltESRawToRecHitFacility,"
    modules += "-hltEcalRegionalJetsRecHit,"
    modules += "-hltEcalRegionalMuonsRecHit,"
    modules += "-hltEcalRegionalEgammaRecHit,"
    modules += "-hltEcalRecHitAll,"
    modules += "-hltESRecHitAll,"
    modules += "-hltL3TauPixelSeeds,"
    modules += "-hltL3TauHighPtPixelSeeds,"
    modules += "-hltL3TauCkfTrackCandidates,"
    modules += "-hltL3TauCkfHighPtTrackCandidates,"
    modules += "-hltL3TauCtfWithMaterialTracks,"
    modules += "-hltL25TauPixelSeeds,"
    modules += "-hltL25TauCkfTrackCandidates,"
    modules += "-hltL25TauCtfWithMaterialTracks,"
    modules += "-hltL3TauSingleTrack15CtfWithMaterialTracks,"
    modules += "-hltPFJetCtfWithMaterialTracks,"
    modules += "-hltBLifetimeRegionalPixelSeedGeneratorStartup,"
    modules += "-hltBLifetimeRegionalCkfTrackCandidatesStartup,"
    modules += "-hltBLifetimeRegionalCtfWithMaterialTracksStartup,"
    modules += "-hltBLifetimeRegionalPixelSeedGeneratorStartupU,"
    modules += "-hltBLifetimeRegionalCkfTrackCandidatesStartupU,"
    modules += "-hltBLifetimeRegionalCtfWithMaterialTracksStartupU,"
    modules += "-hltBLifetimeRegionalPixelSeedGenerator,"
    modules += "-hltBLifetimeRegionalCkfTrackCandidates,"
    modules += "-hltBLifetimeRegionalCtfWithMaterialTracks,"
    modules += "-hltBLifetimeRegionalPixelSeedGeneratorRelaxed,"
    modules += "-hltBLifetimeRegionalCkfTrackCandidatesRelaxed,"
    modules += "-hltBLifetimeRegionalCtfWithMaterialTracksRelaxed,"
    modules += "-hltPixelTracksForMinBias,"
    modules += "-hltPixelTracksForHighMult,"
    modules += "-hltMuonCSCDigis,"
    modules += "-hltMuonDTDigis,"
    modules += "-hltMuonRPCDigis,"
    modules += "-hltGtDigis,"
    modules += "-hltL1GtTrigReport,"
#--- The following modules must always be present to allow for individual paths to be run
    modules += "hltCsc2DRecHits,"
    modules += "hltDt1DRecHits,"
    modules += "hltRpcRecHits"
    
    #--- Some notes about removed sequences ---#
    #--- HLTL1[Non]IsoEgammaRegionalRecoTrackerSequence defined in FastSimulation/EgammaElectronAlgos/data/l1[Non]IsoEgammaRegionalRecoTracker.cff
    #--- HLTL1[Non]Iso[Startup]ElectronsRegionalRecoTrackerSequence defined in FastSimulation/EgammaElectronAlgos/data/l1[Non]IsoElectronsRegionalRecoTracker.cff
    #--- HLTL1[Non]IsoLargeWindowElectronsRegionalRecoTrackerSequence defined in FastSimulation/EgammaElectronAlgos/data/l1[Non]IsoLargeWindowElectronsRegionalRecoTracker.cff
    #--- HLTPixelMatchElectronL1[Non]IsoSequence defined in FastSimulation/EgammaElectronAlgos/data/pixelMatchElectronL1[Non]IsoSequenceForHLT.cff
    #--- HLTPixelMatchElectronL1[Non]IsoLargeWindow[Tracking]Sequence defined in FastSimulation/EgammaElectronAlgos/data/pixelMatchElectronL1[Non]IsoLargeWindowSequenceForHLT.cff
    #--- HLTDoLocal[Pixel/Strip]Sequence defined in FastSimulation/HighLevelTrigger/data/common/HLTSetup.cff
    #--- HLTrecopixelvertexingSequence defined in FastSimulation/Tracking/data/PixelVerticesProducer.cff
    #--- HLTL3displacedMumurecoSequence defined in FastSimulation/HighLevelTrigger/data/btau/L3ForDisplacedMumuTrigger.cff
    #--- HLTPixelTrackingForMinBiasSequence defined in FastSimulation/HighLevelTrigger/data/special/HLTFastRecoForSpecial.cff
    #--- HLTEndSequence defined in FastSimulation/Configuration/test/ExampleWithHLT.cff
    sequences = "--sequences "
    sequences += "-HLTL1IsoEgammaRegionalRecoTrackerSequence,"
    sequences += "-HLTL1NonIsoEgammaRegionalRecoTrackerSequence,"
    sequences += "-HLTL1IsoElectronsRegionalRecoTrackerSequence,"
    sequences += "-HLTL1NonIsoElectronsRegionalRecoTrackerSequence,"
    sequences += "-HLTPixelMatchLargeWindowElectronL1IsoTrackingSequence,"
    sequences += "-HLTPixelMatchLargeWindowElectronL1NonIsoTrackingSequence,"
    sequences += "-HLTPixelTrackingForMinBiasSequence,"
    sequences += "-HLTDoLocalStripSequence,"
    sequences += "-HLTDoLocalPixelSequence,"
    #
    #--- ADAM for muon cascade algo
    #sequences += "-HLTL3muonTkCandidateSequence,"
    #---
    sequences += "-HLTRecopixelvertexingSequence,"
    sequences += "-HLTL3TauTrackReconstructionSequence,"
    sequences += "-HLTL3TauHighPtTrackReconstructionSequence,"
    sequences += "-HLTL25TauTrackReconstructionSequence,"
    sequences += "-HLTL3TauSingleTrack15ReconstructionSequence,"
    sequences += "-HLTTrackReconstructionForJets,"
    sequences += "-HLTEndSequence,"
    sequences += "-HLTBeginSequence,"
    sequences += "-HLTBeginSequenceNZS,"
    sequences += "-HLTBeginSequenceBPTX,"
    sequences += "-HLTBeginSequenceAntiBPTX,"
    sequences += "-HLTL2HcalIsolTrackSequence,"
    sequences += "-HLTL2HcalIsolTrackSequenceHB,"
    sequences += "-HLTL2HcalIsolTrackSequenceHE,"
    sequences += "-HLTL3HcalIsolTrackSequence"

    #--- Some notes about removed paths: ---#
    #--- CandHLTCSCBeamHalo removed because of L1_SingleMuBeamHalo (not found in L1Menu2007)
    #--- HLT1MuonL1Open removed because of L1_SingleMuOpen (not found in L1Menu2007)
    #--- HLTMinBiasHcal removed because no relevant L1 bits found in L1Menu2007
    #--- HLTMinBiasEcal removed because no relevant L1 bits found in L1Menu2007
    #--- HLT4jet30 removed because of L1_QuadJet15 (not found in L1Menu2007)
    #--- HLTXElectron3Jet30 removed because of L1_EG5_TripleJet15 (not found in L1Menu2007)
    #--- CandHLTCSCBeamHaloOverlapRing1 removed because of L1_SingleMuBeamHalo (not found in L1Menu2007)
    #--- CandHLTCSCBeamHaloOverlapRing2 removed because of L1_SingleMuBeamHalo (not found in L1Menu2007)
    #--- CandHLTCSCBeamHaloRing2or3 removed because of L1_SingleMuBeamHalo (not found in L1Menu2007)
    paths = "--paths "

    if L1Menu == "L1Menu_Commissioning2009_v0":
        # remove output endpaths
        paths += "-HLTOutput,"
        paths += "-AlCaOutput,"
        paths += "-AlCaPPOutput,"
        paths += "-AlCaHIOutput,"
        paths += "-ExpressOutput,"
        paths += "-EventDisplayOutput,"
        paths += "-DQMOutput,"
        paths += "-HLTDQMOutput,"
        paths += "-HLTMONOutput,"
        paths += "-HLTDQMResultsOutput,"
        paths += "-NanoDSTOutput,"
        # remove unsupported paths
        paths += "-HLT_HcalPhiSym,"
        paths += "-HLT_Mu0_Track0_Jpsi,"
        paths += "-HLT_Mu3_Track0_Jpsi,"
        paths += "-HLT_Mu3_Track3_Jpsi,"
        paths += "-HLT_Mu3_Track3_Jpsi_v3,"
        paths += "-HLT_Mu3_Track3_Jpsi_v4,"
        paths += "-HLT_Mu3_Track5_Jpsi_v3,"
        paths += "-HLT_Mu3_Track5_Jpsi_v4,"
        paths += "-HLT_Mu5_Track0_Jpsi,"
        paths += "-HLT_Mu5_Track0_Jpsi_v2,"
        paths += "-HLT_Mu5_Track0_Jpsi_v3,"
        paths += "-HLT_Mu0_TkMu0_OST_Jpsi,"
        paths += "-HLT_Mu0_TkMu0_OST_Jpsi_Tight_v3,"
        paths += "-HLT_Mu0_TkMu0_OST_Jpsi_Tight_v4,"
        paths += "-HLT_Mu3_TkMu0_OST_Jpsi,"
        paths += "-HLT_Mu3_TkMu0_OST_Jpsi_Tight_v3,"
        paths += "-HLT_Mu3_TkMu0_OST_Jpsi_Tight_v4,"
        paths += "-HLT_Mu5_TkMu0_OST_Jpsi,"
        paths += "-HLT_Mu5_TkMu0_OST_Jpsi_Tight_v2,"
        paths += "-HLT_Mu5_TkMu0_OST_Jpsi_Tight_v3,"
        paths += "-HLT_Mu5_TkMu0_OST_Jpsi_Tight_v4,"
        paths += "-HLT_L1DoubleMuOpen_Tight,"
        paths += "-HLT_SelectEcalSpikes_L1R,"
        paths += "-HLT_SelectEcalSpikesHighEt_L1R,"
        paths += "-HLT_Ele15_SiStrip_L1R,"
        paths += "-HLT_Ele20_SiStrip_L1R,"
        paths += "-HLT_IsoTrackHB_v2,"
        paths += "-HLT_IsoTrackHE_v2,"
        paths += "-HLT_IsoTrackHB_v3,"
        paths += "-HLT_IsoTrackHE_v3,"
        paths += "-HLT_HcalNZS,"
        paths += "-HLT_Activity_L1A,"
        paths += "-HLT_Activity_PixelClusters,"
        paths += "-HLT_Activity_DT,"
        paths += "-HLT_Activity_DT_Tuned,"
        paths += "-HLT_Activity_Ecal,"
        paths += "-HLT_Activity_EcalREM,"
        paths += "-HLT_Activity_Ecal_SC7,"
        paths += "-HLT_Activity_Ecal_SC15,"
        paths += "-HLT_Activity_Ecal_SC17,"
        paths += "-HLT_DTErrors,"
        paths += "-HLT_HFThreshold3,"
        paths += "-HLT_HFThreshold10,"
        paths += "-HLT_EgammaSuperClusterOnly_L1R,"
        paths += "-HLT_Random,"
        paths += "-HLT_Calibration,"
        paths += "-HLT_EcalCalibration,"
        paths += "-HLT_HcalCalibration,"
        paths += "-AlCa_EcalPi0,"
        paths += "-AlCa_EcalEta,"
###AP *** other paths removed with V01-16-23 HLTrigger/Configuration - ConfDB /dev/CMSSW_3_5_5/XXXX/V21
        paths += "-HLT_DoublePhoton4_eeRes_L1R,"
        paths += "-HLT_DoublePhoton4_Jpsi_L1R,"
        paths += "-HLT_DoublePhoton4_Upsilon_L1R,"
        paths += "-HLT_DoubleEle5_SW_Upsilon_L1R_v1,"
        paths += "-HLT_DoubleEle5_SW_Upsilon_L1R_v2,"
        paths += "-DQM_FEDIntegrity,"
        paths += "-DQM_FEDIntegrity_v2,"
        paths += "-AlCa_EcalPhiSym,"
###SA *** removed with V01-17-07 HLTrigger/Configuration - ConfDB /dev/CMSSW_3_6_0/pre4/XXXX/V13
        paths += "-HLT_L1MuOpen_AntiBPTX,"
        paths += "-HLT_L1MuOpen_AntiBPTX_v2,"
###SA *** removed with V01-17-29 HLTrigger/Configuration - ConfDB /dev/CMSSW_3_6_0/GRun/V15
        paths += "-HLT_Jet15U_HcalNoiseFiltered,"
        paths += "-HLT_Jet15U_HcalNoiseFiltered_v3,"
###AP *** remove useless (and trouble prone) AlCa paths:
        paths += "-AlCa_RPCMuonNoHits,"
        paths += "-AlCa_RPCMuonNoTriggers,"
        paths += "-AlCa_RPCMuonNormalisation,"
###AP *** remove on 13/08/2010 (3.5E30 menu):
        paths += "-HLT_DoubleEle4_SW_eeRes_L1R,"
        paths += "-HLT_DoubleEle4_SW_eeRes_L1R_v2,"
###AP *** remove on 08/09/2010 (2E31 menu)
        paths += "-HLT_Mu3_Track3_Jpsi,"
###AP *** save cpu time by removing HLTAnalyzerEndpath
        paths += "-HLTAnalyzerEndpath,"
###
        paths += "-DummyPath"


    #--- Special case: Running a user-specified set of paths ---#
    if usePaths != "All":
        paths = "--paths " + usePaths

    services = "--services -PrescaleService,-MessageLogger,-DQM,-FUShmDQMOutputService,-MicroStateService,-ModuleWebRegistry,-TimeProfilerService,-UpdaterService"

    psets = "--psets "
    psets += "-options,"
    psets += "-maxEvents"

    #--- Adapt for python ---#
    #subStart = cffName.find(".")
    #subEnd = len(cffName)
    #cffType = cffName[subStart:subEnd]
    if cffName.endswith(".py"):
        cffType = ".py"
    else:
        cffType = ".cff"

    baseCommand = "edmConfigFromDB --cff --configName " + dbName
    if cffType == ".py":
        baseCommand += " --format Python"

    # myGetCff = "edmConfigFromDB --cff --configName " + dbName + " " + essources + " " + esmodules + " " + services + " " + psets + " > " + cffName
    if ( blockName == "None" ):
        myGetCff = baseCommand + " " + essources + " " + esmodules + " " + blocks + " " + sequences + " " + modules + " " + paths + " " + services + " " + psets + " > " + cffName
    else:
        blockpaths = "--nopaths"
        blockessources = "--noessources"
        blockesmodules = "--noesmodules"
        blockservices = "--noservices"
        blockpsets = "--nopsets"
        myGetCff = baseCommand + " " + essources + " " + esmodules + " " + sequences + " " + modules + " " + paths + " " + services + " " + psets + " > " + cffName
        myGetBlocks = baseCommand + " " + blockessources + " " + blockesmodules + " " + blockservices + " " + blockpaths + " " + blockpsets + " " + blocks + " > " + blockName

    # Write blocks for electrons and muons, in py configuration
    if ( blockName != "None" ) :
        os.system(myGetBlocks)

        # online to offline conversion - taken from HLTrigger/Configuration/test/getHLT.py
        # FIXME these should be better integrated with edmConfigFromDB
        os.system("sed -e'/^streams/,/^)/d' -e'/^datasets/,/^)/d' -i %s" % blockName)

        bName = "None"
        for line in fileinput.input(blockName,inplace=1):

            if line.find("hltOfflineBeamSpot") > 0:
                line = line.replace('hltOfflineBeamSpot','offlineBeamSpot')

            if line.find("block_hlt") == 0:
                subStart = line.find("_")
                subEnd = line.find(" ",subStart)
                bName = line[subStart:subEnd]

            if line.find("SeedConfiguration") == 0:
                if bName == "None":
                    print line[:-1]
            elif line.find("MuonTrackingRegionBuilder") == 0:
                if bName == "None":
                    print line[:-1]
            elif line.find(")") == 0:
                if bName != "None":
                    bName = "None"
                else:
                    print line[:-1]
            else:
                print line[:-1]

    # Write all HLT
    os.system(myGetCff)

    # online to offline conversion - taken from HLTrigger/Configuration/test/getHLT.py
    # FIXME these should be better integrated with edmConfigFromDB
    os.system("sed -e'/^streams/,/^)/d' -e'/^datasets/,/^)/d' -i %s" % cffName)
    os.system("sed -e 's/cms.InputTag( \"source\" )/cms.InputTag( \"rawDataCollector\" )/' -i %s" % cffName)
    # FIXME - DTUnpackingModule should not have untracked parameters
    # os.system("sed -e'/DTUnpackingModule/a\ \ \ \ inputLabel = cms.untracked.InputTag( \"rawDataCollector\" ),' -i %s" % cffName)

    # myReplaceTrigResults = "replace TriggerResults::HLT " + process + " -- " + cffName
    # os.system(myReplaceTrigResults)

    # Make replace statements at the beginning of the cff
    mName = "None"
    mType = "None"
    bName = "None"
    #schedule = "process.schedule = cms.Schedule(process."
    for line in fileinput.input(cffName,inplace=1):
        if cffType == ".py":
            if line.find("cms.EDProducer") >= 0:
                subStart = line.find("hlt")
                subEnd = line.find(" ",subStart)
                mName = line[subStart:subEnd]
                subStart = line.find("=") + 19
                subEnd = line.find(" ",subStart)
                mType = line[subStart:subEnd-2]
            #if line.find("cms.Path") > 0:
                #subEnd = line.find("cms.Path")
                #myPath = line[0:subEnd-3]
                #schedule += myPath
                #if myPath != "HLTriggerFinalPath":
                    #schedule += ",process."
                #else:
                    #schedule += ")"
        else:
            if line.find("module hlt") >= 0:
                subStart = line.find("hlt")
                subEnd = line.find(" ",subStart)
                mName = line[subStart:subEnd]
                subStart = line.find("=") + 2
                subEnd = line.find(" ",subStart)
                mType = line[subStart:subEnd]

        if line.find("block block_hlt") == 0:
            subStart = line.find("_")
            subEnd = line.find(" ",subStart)
            bName = line[subStart:subEnd]

##HLTL3PixelIsolFilterSequence has been removed
        if line.find("HLTL3PixelIsolFilterSequence = ") == 0:
            line = line.replace('hltPixelTracks','hltPixelTracking')
            print line[:-1]
        elif line.find("HLTRecopixelvertexingForMinBiasSequence = ") == 0:
            line = line.replace('hltPixelTracksForMinBias','pixelTripletSeedsForMinBias*hltPixelTracksForMinBias')
            print line[:-1]
        elif line.find("GMTReadoutCollection") > 0:
            if mName == "hltL2MuonSeeds":
                line = line.replace('hltGtDigis','gmtDigis')
            print line[:-1]
        elif line.find("InputObjects") > 0:
            if mName == "hltL2MuonSeeds":
                line = line.replace('hltL1extraParticles','l1extraParticles')
            print line[:-1]
        elif line.find("L1MuonCollectionTag") > 0:
            line = line.replace('hltL1extraParticles','l1extraParticles')
            print line[:-1]
        elif line.find("L1CollectionsTag") > 0:
            line = line.replace('hltL1extraParticles','l1extraParticles')
            print line[:-1]
        elif line.find("L1GtObjectMapTag") > 0:
            line = line.replace('hltL1GtObjectMap','gtDigis')
            print line[:-1]
        elif line.find("L1GtReadoutRecordTag") > 0:
            line = line.replace('hltGtDigis','gtDigis')
            print line[:-1]
        elif line.find("PSet SeedConfiguration") > 0:
            if bName == "None":
                print line[:-1]
        elif line.find("PSet MuonTrackingRegionBuilder") > 0:
            if bName == "None":
                print line[:-1]
        elif line.find("CandTag") > 0:
            line = line.replace('hltL1extraParticles','l1extraParticles')
            print line[:-1]
        elif line.find("preFilteredSeeds") > 0:
            line = line.replace('True','False')
            print line[:-1]
        elif line.find("initialSeeds") > 0:
            line = line.replace('noSeedsHere','globalPixelSeeds:GlobalPixel')
            print line[:-1]
        elif line.find("}") > 0:
            if bName != "None":
                bName = "None"
            else:
                print line[:-1]
        elif line.find("hltMeasurementTracker") > 0:
            line = line.replace('hltMeasurementTracker', '')
            print line[:-1]
        elif line.find("hltCkfTrajectoryBuilder") > 0:
            line = line.replace('hltCkfTrajectoryBuilder', 'CkfTrajectoryBuilder')
            print line[:-1]
        else:
            print line[:-1]
            # The first line should be where the comments go
            if line.find("//") == 0 or line.find("#") == 0:
                print "# Begin replace statements specific to the FastSim HLT"
                print "# For all HLTLevel1GTSeed objects, make the following replacements:"
                print "#   - L1GtReadoutRecordTag changed from hltGtDigis to gtDigis"
                print "#   - L1CollectionsTag changed from hltL1extraParticles to l1extraParticles"
                print "#   - L1MuonCollectionTag changed from hltL1extraParticles to l1extraParticles"
                print "# For hltL2MuonSeeds: "
                print "#   - InputObjects changed from hltL1extraParticles to l1extraParticles"
                print "#   - GMTReadoutCollection changed from hltGtDigis to gmtDigis"
                print "# All other occurances of hltL1extraParticles recast as l1extraParticles"
                print "# L1GtObjectMapTag: hltL1GtObjectMap recast as gtDigis"
                print "# L1GtReadoutRecordTag: hltGtDigis recast as gtDigis"
                print "# hltMuon[CSC/DT/RPC]Digis changed to muon[CSC/DT/RPC]Digis"
                print "# Replace hltOfflineBeamSpot with offlineBeamSpot"
                print "# AlCaIsoTrack needs HLTpixelTracking instead of pixelTracks"
                print "# Some HLT modules were recast as FastSim sequences: "
                print "#   - hltL3TrackCandidateFromL2, see FastSimulation/HighLevelTrigger/data/Muon/HLTFastRecoForMuon.cff"
                print "#   - hltCkfTrackCandidatesL3SingleTau[MET][Relaxed], see FastSimulation/HighLevelTrigger/data/btau/HLTFastRecoForTau.cff"
                print "#   - hltCkfTrackCandidatesMumu, see FastSimulation/HighLevelTrigger/data/btau/L3ForDisplacedMumuTrigger.cff"
                print "#   - hltCkfTrackCandidatesMumuk, see FastSimulation/HighLevelTrigger/data/btau/L3ForMuMuk.cff"
                print "#   - hltBLifetimeRegionalCkfTrackCandidates[Relaxed], see FastSimulation/HighLevelTrigger/data/btau/lifetimeRegionalTracking.cff"
                print "# See FastSimulation/Configuration/test/getFastSimHLT_8E29_cff.py for other documentation"
                print "# (L1Menu2007 only) Replace L1_QuadJet30 with L1_QuadJet40"
                print "# (Temporary) Remove PSet begin and end from block"
                print "# End replace statements specific to the FastSim HLT"
            if cffType == ".py":
                if line.find("#") == 0:
                    print "# Additional import to make this file self contained"
                    print "from FastSimulation.HighLevelTrigger.HLTSetup_cff import *"
                #if line.find("cms.EndPath") > 0:
                    #print schedule

    # Now try to make the replacements
    for line in fileinput.input(cffName,inplace=1):
        if line.find("//") < 0:
            if line.find("hltOfflineBeamSpot") > 0:
                line = line.replace('hltOfflineBeamSpot','offlineBeamSpot')
            if line.find("hltMuonCSCDigis") > 0:
                if cffType == ".py":
                    if line.find("hltMuonCSCDigis +") > 0:
                        line = line.replace('hltMuonCSCDigis','cms.SequencePlaceholder("simMuonCSCDigis")')
                    else:
                        line = line.replace('hltMuonCSCDigis','simMuonCSCDigis')
                else:
                    line = line.replace('hltMuonCSCDigis','simMuonCSCDigis')
            if line.find("hltMuonDTDigis") > 0:
                if cffType == ".py":
                    if line.find("hltMuonDTDigis +") > 0:
                        line = line.replace('hltMuonDTDigis','cms.SequencePlaceholder("simMuonDTDigis")')
                    else:
                        line = line.replace('hltMuonDTDigis','simMuonDTDigis')
                else:
                    line = line.replace('hltMuonDTDigis','simMuonDTDigis')
            if line.find("hltMuonRPCDigis") > 0:
                if cffType == ".py":
                    if line.find("hltMuonRPCDigis +") > 0:
                        line = line.replace('hltMuonRPCDigis','cms.SequencePlaceholder("simMuonRPCDigis")')
                    else:
                        line = line.replace('hltMuonRPCDigis','simMuonRPCDigis')
                else:
                    line = line.replace('hltMuonRPCDigis','simMuonRPCDigis')
            if line.find("HLTEndSequence") > 0:
                if cffType == ".py":
                    line = line.replace('HLTEndSequence','cms.SequencePlaceholder("HLTEndSequence")')
            if line.find("hltL1extraParticles") > 0:
                line = line.replace('hltL1extraParticles','l1extraParticles')
            if line.find("QuadJet30") > 0:
                if L1Menu == "L1Menu2007":
                    line = line.replace('QuadJet30','QuadJet40')
            if line.find("hltL1IsoLargeWindowElectronPixelSeeds") > 0:
                if line.find("Sequence") > 0:
                    line = line.replace("hltL1IsoLargeWindowElectronPixelSeeds","hltL1IsoLargeWindowElectronPixelSeedsSequence")
            if line.find("hltL1NonIsoLargeWindowElectronPixelSeeds") > 0:
                if line.find("Sequence") > 0:
                    line = line.replace("hltL1NonIsoLargeWindowElectronPixelSeeds","hltL1NonIsoLargeWindowElectronPixelSeedsSequence")
            if line.find("hltL1IsoEgammaRegionalPixelSeedGenerator + hltL1IsoEgammaRegionalCkfTrackCandidates + hltL1IsoEgammaRegionalCTFFinalFitWithMaterial") > 0:
                line = line.replace('hltL1IsoEgammaRegionalPixelSeedGenerator + hltL1IsoEgammaRegionalCkfTrackCandidates + hltL1IsoEgammaRegionalCTFFinalFitWithMaterial', 'HLTL1IsoEgammaRegionalRecoTrackerSequence')
            if line.find("hltL1NonIsoEgammaRegionalPixelSeedGenerator + hltL1NonIsoEgammaRegionalCkfTrackCandidates + hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial") > 0:
                line = line.replace('hltL1NonIsoEgammaRegionalPixelSeedGenerator + hltL1NonIsoEgammaRegionalCkfTrackCandidates + hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial', 'HLTL1NonIsoEgammaRegionalRecoTrackerSequence')
            if line.find("cms.Path") > 0:
                line = line.replace('hltGtDigis', 'HLTBeginSequence')

        print line[:-1]

