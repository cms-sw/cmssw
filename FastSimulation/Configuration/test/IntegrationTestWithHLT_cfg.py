import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Generate ttbar events
process.load("FastSimulation.Configuration.ttbar_cfi")

# L1 Menu and prescale factors : useful for testing all L1 paths
process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")
# Other choices are
# L1 Menu 2008 2x10E30 - Prescale
# process.load("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/lumi1030/L1Menu2008_2E30.cff")
# L1 Menu 2008 2x10E30 - No Prescale
# process.load("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/lumi1030/L1Menu2008_2E30_Unprescaled.cff")
# L1 Menu 2008 2x10E31 - Prescale
# process.load("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/lumi1031/L1Menu2008_2E31.cff")
# L1 Menu 2008 2x10E31 - No Prescale
# process.load("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/lumi1031/L1Menu2008_2E31_Unprescaled.cff")
# L1 Menu 2008 10E32 - Prescale 
# process.load("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/lumix1032/L1Menu2007.cff")
# L1 Menu 2008 10E32 - No Prescale 
# process.load("L1TriggerConfig/L1GtConfigProducers/data/Luminosity/lumix1032/L1Menu2007_Unprescaled.cff")

# Common inputs, with fake conditions
process.load("FastSimulation.Configuration.CommonInputs_cff")

# L1 Emulator and HLT Setup
process.load("FastSimulation.HighLevelTrigger.HLTSetup_cff")

# Famos sequences
process.load("FastSimulation.Configuration.FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# HLT paths - defined by configDB
# This one is created on the fly by FastSimulation/Configuration/test/IntegrationTestWithHLT_py.csh
process.load("FastSimulation.Configuration.HLT_cff")

# Only event accepted by L1 + HLT are reconstructed
process.HLTEndSequence = cms.Sequence(process.reconstructionWithFamos)

# No reconstruction - only HLT (needed also for reconstructing all events)
# process.HLTEndSequence = cms.Sequence(process.dummyModule)

# Schedule the HLT paths
process.schedule = cms.Schedule(process.HLTriggerFirstPath,process.HLT_L1Jet15,process.HLT_Jet30,process.HLT_Jet50,process.HLT_Jet80,process.HLT_Jet110,process.HLT_Jet180,process.HLT_Jet250,process.HLT_FwdJet20,process.HLT_DoubleJet150,process.HLT_DoubleJet125_Aco,process.HLT_DoubleFwdJet50,process.HLT_DiJetAve15,process.HLT_DiJetAve30,process.HLT_DiJetAve50,process.HLT_DiJetAve70,process.HLT_DiJetAve130,process.HLT_DiJetAve220,process.HLT_TripleJet85,process.HLT_QuadJet30,process.HLT_QuadJet60,process.HLT_SumET120,process.HLT_L1MET20,process.HLT_MET25,process.HLT_MET35,process.HLT_MET50,process.HLT_MET65,process.HLT_MET75,process.HLT_MET35_HT350,process.HLT_Jet180_MET60,process.HLT_Jet60_MET70_Aco,process.HLT_Jet100_MET60_Aco,process.HLT_DoubleJet125_MET60,process.HLT_DoubleFwdJet40_MET60,process.HLT_DoubleJet60_MET60_Aco,process.HLT_DoubleJet50_MET70_Aco,process.HLT_DoubleJet40_MET70_Aco,process.HLT_Jet80_Jet20_MET60_NV,process.HLT_TripleJet60_MET60,process.HLT_QuadJet35_MET60,process.HLT_IsoEle15_L1I,process.HLT_IsoEle18_L1R,process.HLT_IsoEle12_SW_L1R,process.HLT_IsoEle15_LW_L1I,process.HLT_IsoEle18_LW_L1R,process.HLT_LooseIsoEle15_SW_L1R,process.HLT_LooseIsoEle18_LW_L1R,process.HLT_LooseIsoEle15_LW_L1R,process.HLT_Ele8_SW_L1R,process.HLT_Ele10_SW_L1R,process.HLT_Ele15_SW_L1R,process.HLT_Ele18_SW_L1R,process.HLT_Ele12_LW_L1R,process.HLT_Ele15_LW_L1R,process.HLT_EM80,process.HLT_EM200,process.HLT_DoubleIsoEle10_L1I,process.HLT_DoubleIsoEle12_L1R,process.HLT_DoubleIsoEle10_LW_L1I,process.HLT_DoubleIsoEle12_LW_L1R,process.HLT_DoubleEle5_SW_L1R,process.HLT_DoubleEle8_LW_OnlyPixelM_L1R,process.HLT_DoubleEle10_LW_OnlyPixelM_L1R,process.HLT_DoubleEle12_LW_OnlyPixelM_L1R,process.HLT_DoubleEle10_Z,process.HLT_DoubleEle6_Exclusive,process.HLT_IsoPhoton12_L1I,process.HLT_IsoPhoton30_L1I,process.HLT_IsoPhoton10_L1R,process.HLT_IsoPhoton15_L1R,process.HLT_IsoPhoton20_L1R,process.HLT_IsoPhoton25_L1R,process.HLT_IsoPhoton40_L1R,process.HLT_LooseIsoPhoton20_L1R,process.HLT_LooseIsoPhoton30_L1R,process.HLT_LooseIsoPhoton40_L1R,process.HLT_LooseIsoPhoton45_L1R,process.HLT_Photon15_L1R,process.HLT_Photon25_L1R,process.HLT_Photon30_L1R,process.HLT_Photon40_L1R,process.HLT_DoubleIsoPhoton20_L1I,process.HLT_DoubleIsoPhoton20_L1R,process.HLT_DoubleLooseIsoPhoton20_L1R,process.HLT_DoublePhoton8_L1R,process.HLT_DoublePhoton10_L1R,process.HLT_DoublePhoton20_L1R,process.HLT_DoublePhoton10_Exclusive,process.HLT_L1Mu,process.HLT_L1MuOpen,process.HLT_L2Mu9,process.HLT_IsoMu9,process.HLT_IsoMu11,process.HLT_IsoMu13,process.HLT_IsoMu15,process.HLT_NoTrackerIsoMu15,process.HLT_Mu3,process.HLT_Mu5,process.HLT_Mu7,process.HLT_Mu9,process.HLT_Mu10,process.HLT_Mu11,process.HLT_Mu13,process.HLT_Mu15,process.HLT_Mu15_L1Mu7,process.HLT_Mu15_Vtx2cm,process.HLT_Mu15_Vtx2mm,process.HLT_DoubleIsoMu3,process.HLT_DoubleMu3,process.HLT_DoubleMu3_Vtx2cm,process.HLT_DoubleMu3_Vtx2mm,process.HLT_DoubleMu3_JPsi,process.HLT_DoubleMu3_Upsilon,process.HLT_DoubleMu7_Z,process.HLT_DoubleMu3_SameSign,process.HLT_DoubleMu3_Psi2S,process.HLT_TripleMu3,process.HLT_BTagIP_Jet180,process.HLT_BTagIP_Jet120_Relaxed,process.HLT_BTagIP_Jet160_Relaxed,process.HLT_BTagIP_DoubleJet120,process.HLT_BTagIP_DoubleJet60_Relaxed,process.HLT_BTagIP_DoubleJet100_Relaxed,process.HLT_BTagIP_TripleJet70,process.HLT_BTagIP_TripleJet60_Relaxed,process.HLT_BTagIP_TripleJet40_Relaxed,process.HLT_BTagIP_QuadJet40,process.HLT_BTagIP_QuadJet35_Relaxed,process.HLT_BTagIP_QuadJet30_Relaxed,process.HLT_BTagIP_HT470,process.HLT_BTagIP_HT420_Relaxed,process.HLT_BTagIP_HT320_Relaxed,process.HLT_BTagMu_DoubleJet120,process.HLT_BTagMu_DoubleJet100_Relaxed,process.HLT_BTagMu_DoubleJet60_Relaxed,process.HLT_BTagMu_TripleJet70,process.HLT_BTagMu_TripleJet60_Relaxed,process.HLT_BTagMu_TripleJet40_Relaxed,process.HLT_BTagMu_QuadJet40,process.HLT_BTagMu_QuadJet35_Relaxed,process.HLT_BTagMu_QuadJet30_Relaxed,process.HLT_BTagMu_HT370,process.HLT_BTagMu_HT330_Relaxed,process.HLT_BTagMu_HT250_Relaxed,process.HLT_DoubleMu3_BJPsi,process.HLT_DoubleMu4_BJPsi,process.HLT_TripleMu3_TauTo3Mu,process.HLT_IsoTau_MET65_Trk20,process.HLT_IsoTau_MET35_Trk15_L1MET,process.HLT_LooseIsoTau_MET30,process.HLT_LooseIsoTau_MET30_L1MET,process.HLT_DoubleIsoTau_Trk3,process.HLT_DoubleLooseIsoTau,process.HLT_IsoEle8_IsoMu7,process.HLT_IsoEle10_Mu10_L1R,process.HLT_IsoEle12_IsoTau_Trk3,process.HLT_IsoEle10_BTagIP_Jet35,process.HLT_IsoEle12_Jet40,process.HLT_IsoEle12_DoubleJet80,process.HLT_IsoElec5_TripleJet30,process.HLT_IsoEle12_TripleJet60,process.HLT_IsoEle12_QuadJet35,process.HLT_IsoMu14_IsoTau_Trk3,process.HLT_IsoMu7_BTagIP_Jet35,process.HLT_IsoMu7_BTagMu_Jet20,process.HLT_IsoMu7_Jet40,process.HLT_NoL2IsoMu8_Jet40,process.HLT_Mu14_Jet50,process.HLT_Mu5_TripleJet30,process.HLT_BTagMu_Jet20_Calib,process.HLT_ZeroBias,process.HLT_MinBias,process.HLT_MinBiasHcal,process.HLT_MinBiasEcal,process.HLT_MinBiasPixel,process.HLT_MinBiasPixel_Trk5,process.HLT_BackwardBSC,process.HLT_ForwardBSC,process.HLT_CSCBeamHalo,process.HLT_CSCBeamHaloOverlapRing1,process.HLT_CSCBeamHaloOverlapRing2,process.HLT_CSCBeamHaloRing2or3,process.HLT_TrackerCosmics,process.HLT_TriggerType,process.AlCa_IsoTrack,process.AlCa_EcalPhiSym,process.AlCa_EcalPi0,process.HLTriggerFinalPath)

# All events are reconstructed, including those rejected at L1/HLT
# reconstruction = cms.Path( reconstructionWithFamos )
# process.schedule.append(reconstruction)

# Simulation sequence
process.simulation = cms.Sequence(process.simulationWithFamos)
# You many not want to simulate everything
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
# Parameterized magnetic field
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
# Number of pileup events per crossing
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

# Get frontier conditions   - not applied in the HCAL, see below
# Values for globaltag are "STARTUP_V1::All", "1PB::All", "10PB::All", "IDEAL_V2::All"
process.GlobalTag.globaltag = "STARTUP_V1::All"

# Apply ECAL and HCAL miscalibration 
process.caloRecHits.RecHitsFactory.doMiscalib = True

# Apply Tracker misalignment (ideal alignment though)
process.famosSimHits.ApplyAlignment = True
process.misalignedTrackerGeometry.applyAlignment = True

# Apply HCAL miscalibration (not ideal in that case). Choose between hcalmiscalib_startup.xml , hcalmiscalib_1pb.xml , hcalmiscalib_10pb.xml (startup is the default)
process.caloRecHits.RecHitsFactory.HCAL.Refactor = 1.0
process.caloRecHits.RecHitsFactory.HCAL.Refactor_mean = 1.0
#process.caloRecHits.RecHitsFactory.HCAL.fileNameHcal = "hcalmiscalib_startup.xml"


# Note : if your process is not called HLT, you have to change that! 
# process.hltTrigReport.HLTriggerResults = TriggerResults::PROD
# process.hltHighLevel.TriggerResultsTag = TriggerResults::PROD 

# To write out events 
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('AODIntegrationTestWithHLT.root')
)
process.outpath = cms.EndPath(process.o1)

# Add endpaths to the schedule
process.schedule.append(process.HLTAnalyzerEndpath)
process.schedule.append(process.outpath)

# Keep the logging output to a nice level #
#process.Timing =  cms.Service("Timing")
#process.load("FWCore/MessageService/MessageLogger_cfi")
#process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt")


