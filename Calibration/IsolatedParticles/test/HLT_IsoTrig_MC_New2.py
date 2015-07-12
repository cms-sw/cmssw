import os
import FWCore.ParameterSet.Config as cms

process = cms.Process('HLTNew1')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('CommonTools.ParticleFlow.EITopPAG_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(90)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
        #        'file:step2.root')
        "file:/afs/cern.ch/work/r/ruchi/public/0048FA62-B924-E411-A09C-002590DB916E.root"),
                            skipEvents=cms.untracked.uint32(190)
                            )

process.options = cms.untracked.PSet(
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications')
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('Try_HLTIsoTrig_MC_New1_2.root')
                                   )

process.load('Calibration.IsolatedParticles.isoTrig_cfi')
process.IsoTrigHB.Verbosity = 0
process.IsoTrigHB.ProcessName = "HLTNew1"
process.IsoTrigHB.DoL2L3 = True
process.IsoTrigHB.DoTimingTree = False
process.IsoTrigHB.DoStudyIsol = True
process.IsoTrigHB.DoChgIsolTree = True
process.IsoTrigHB.DoMipCutTree = False

process.IsoTrigHE = process.IsoTrigHB.clone(
    Triggers       = ["HLT_IsoTrackHE"],
    DoTimingTree = False)

process.load('Calibration.IsolatedParticles.isoTrackCalibration_cfi')
process.IsoTrackCalibration.Triggers = ["HLT_IsoTrackHE_v16", "HLT_IsoTrackHB_v15"]
process.IsoTrackCalibration.ProcessName = "HLTNew1"
process.IsoTrackCalibration.L1Filter  = "hltL1sL1SingleJet"
process.IsoTrackCalibration.L2Filter  = "hltIsolPixelTrackL2Filter"
process.IsoTrackCalibration.L3Filter  = "L3Filter"
process.IsoTrackCalibration.Verbosity = 0

process.analyze = cms.EndPath(process.IsoTrigHB + process.IsoTrigHE + process.IsoTrackCalibration)

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

process.load('Calibration.IsolatedParticles.HLT_IsoTrack_cff')

process.HLT_IsoTrackHE_v16 = cms.Path(process.HLTBeginSequence + 
                                      process.hltL1sL1SingleJet68 + 
                                      process.hltPreIsoTrackHE +
                                      process.HLTDoLocalPixelSequence + 
                                      process.hltPixelLayerTriplets + 
                                      process.hltPixelTracks + 
                                      process.hltPixelVertices +
                                      process.hltTrimmedPixelVertices +
                                      process.hltIsolPixelTrackProdHE +
                                      process.hltIsolPixelTrackL2FilterHE + 
                                      process.HLTDoFullUnpackingEgammaEcalSequence + 
                                      process.hltIsolEcalPixelTrackProdHE + 
                                      process.hltEcalIsolPixelTrackL2FilterHE +
                                      process.HLTDoLocalStripSequence + 
                                      process.hltIter0PFLowPixelSeedsFromPixelTracks +
                                      process.hltIter0PFlowCkfTrackCandidates +
                                      process.hltIter0PFlowCtfWithMaterialTracks +
                                      process.hltHcalITIPTCorrectorHE + 
                                      process.hltIsolPixelTrackL3FilterHE + 
                                      process.HLTEndSequence 
                                      )

process.HLT_IsoTrackHB_v15 = cms.Path(process.HLTBeginSequence + 
                                      process.hltL1sL1SingleJet68 + 
                                      process.hltPreIsoTrackHB +
                                      process.HLTDoLocalPixelSequence + 
                                      process.hltPixelLayerTriplets + 
                                      process.hltPixelTracks + 
                                      process.hltPixelVertices + 
                                      process.hltTrimmedPixelVertices +
                                      process.hltIsolPixelTrackProdHB + 
                                      process.hltIsolPixelTrackL2FilterHB +
                                      process.HLTDoFullUnpackingEgammaEcalSequence + 
                                      process.hltIsolEcalPixelTrackProdHB + 
                                      process.hltEcalIsolPixelTrackL2FilterHB +
                                      process.HLTDoLocalStripSequence + 
                                      process.hltIter0PFLowPixelSeedsFromPixelTracks +
                                      process.hltIter0PFlowCkfTrackCandidates +
                                      process.hltIter0PFlowCtfWithMaterialTracks +
                                      process.hltHcalITIPTCorrectorHB + 
                                      process.hltIsolPixelTrackL3FilterHB + 
                                      process.HLTEndSequence 
                                      )

process.HLTSchedule = cms.Schedule( *(process.HLTriggerFirstPath,
                                      process.HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v1,
                                      process.HLT_DoubleMu33NoFiltersNoVtx_v1,
                                      process.HLT_DoubleMu38NoFiltersNoVtx_v1,
                                      process.HLT_DoubleMu23NoFiltersNoVtxDisplaced_v1,
                                      process.HLT_DoubleMu28NoFiltersNoVtxDisplaced_v1,
                                      process.HLT_Ele20WP60_Ele8_Mass55_v1,
                                      process.HLT_Ele25WP60_SC4_Mass55_v1,
                                      process.HLT_IsoMu17_eta2p1_v1,
                                      process.HLT_DoubleIsoMu17_eta2p1_v1,
                                      process.HLT_IsoMu24_eta2p1_LooseIsoPFTau20_v1,
                                      process.HLT_IsoMu20_eta2p1_CentralPFJet30_BTagCSV_v1,
                                      process.HLT_IsoMu20_eta2p1_TriCentralPFJet40_v1,
                                      process.HLT_IsoMu20_eta2p1_TriCentralPFJet60_50_35_v1,
                                      process.HLT_IsoMu20_eta2p1_v1,process.HLT_IsoMu24_v1,
                                      process.HLT_IsoMu24_eta2p1_CentralPFJet30_BTagCSV_v1,
                                      process.HLT_IsoMu24_eta2p1_TriCentralPFJet40_v1,
                                      process.HLT_IsoMu24_eta2p1_TriCentralPFJet60_50_35_v1,
                                      process.HLT_IsoMu24_eta2p1_v1,
                                      process.HLT_IsoTkMu20_eta2p1_v1,
                                      process.HLT_IsoTkMu24_v1,
                                      process.HLT_IsoTkMu24_eta2p1_v1,
                                      process.HLT_JetE30_NoBPTX3BX_NoHalo_v1,
                                      process.HLT_JetE30_NoBPTX_v1,
                                      process.HLT_JetE50_NoBPTX3BX_NoHalo_v1,
                                      process.HLT_JetE70_NoBPTX3BX_NoHalo_v1,
                                      process.HLT_L2DoubleMu23_NoVertex_v1,
                                      process.HLT_L2DoubleMu28_NoVertex_2Cha_Angle2p5_Mass10_v1,
                                      process.HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_Mass10_v1,
                                      process.HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v1,
                                      process.HLT_L2Mu10_NoVertex_NoBPTX_v1,
                                      process.HLT_L2Mu20_NoVertex_3Sta_NoBPTX3BX_NoHalo_v1,
                                      process.HLT_L2Mu30_NoVertex_3Sta_NoBPTX3BX_NoHalo_v1,
                                      process.HLT_Mu17_Mu8_DZ_v1,
                                      process.HLT_Mu17_TkMu8_DZ_v1,
                                      process.HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v1,
                                      process.HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v1,
                                      process.HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v1,
                                      process.HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v1,
                                      process.HLT_Mu40_eta2p1_PFJet200_PFJet50_v1,
                                      process.HLT_Mu24_eta2p1_v1,
                                      process.HLT_TkMu24_eta2p1_v1,
                                      process.HLT_PFHT350_PFMET120_NoiseCleaned_v1,
                                      process.HLT_PFHT550_4Jet_v1,
                                      process.HLT_PFHT650_4Jet_v1,
                                      process.HLT_PFHT750_4Jet_v1,
                                      process.HLT_PFHT350_v1,
                                      process.HLT_PFHT600_v1,
                                      process.HLT_PFHT650_v1,
                                      process.HLT_PFHT900_v1,
                                      process.HLT_PFJet40_v1,
                                      process.HLT_PFJet60_v1,
                                      process.HLT_PFJet80_v1,
                                      process.HLT_PFJet140_v1,
                                      process.HLT_PFJet200_v1,
                                      process.HLT_DiPFJetAve30_HFJEC_v1,
                                      process.HLT_DiPFJetAve60_HFJEC_v1,
                                      process.HLT_DiPFJetAve80_HFJEC_v1,
                                      process.HLT_DiPFJetAve100_HFJEC_v1,
                                      process.HLT_DiPFJetAve160_HFJEC_v1,
                                      process.HLT_DiPFJet40_DEta3p5_MJJ600_PFMETNoMu80_v1,
                                      process.HLT_HT200_v1,
                                      process.HLT_HT250_v1,
                                      process.HLT_HT300_v1,
                                      process.HLT_HT350_v1,
                                      process.HLT_HT400_v1,
                                      process.HLT_HT200_DiJet90_AlphaT0p57_v1,
                                      process.HLT_HT250_DiJet90_AlphaT0p55_v1,
                                      process.HLT_HT300_DiJet90_AlphaT0p53_v1,
                                      process.HLT_HT350_DiJet90_AlphaT0p52_v1,
                                      process.HLT_HT400_DiJet90_AlphaT0p51_v1,
                                      process.HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_PFMET40_v1,
                                      process.HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_VBF_v1,
                                      process.HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_PFMET40_v1,
                                      process.HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_VBF_v1,
                                      process.HLT_Mu17_TrkIsoVVL_v1,
                                      process.HLT_Mu24_TrkIsoVVL_v1,
                                      process.HLT_Mu34_TrkIsoVVL_v1,
                                      process.HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v1,
                                      process.HLT_Ele33_CaloIdL_TrackIdL_IsoVL_PFJet30_v1,
                                      process.HLT_BTagMu_DiJet20_Mu5_v1,
                                      process.HLT_BTagMu_DiJet40_Mu5_v1,
                                      process.HLT_BTagMu_DiJet70_Mu5_v1,
                                      process.HLT_BTagMu_DiJet110_Mu5_v1,
                                      process.HLT_Ele23_CaloIdL_TrackIdL_IsoVL_v1,
                                      process.HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v1,
                                      process.HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v1,
                                      process.HLT_Photon22_v1,
                                      process.HLT_Photon36_v1,
                                      process.HLT_Photon22_R9Id90_HE10_Iso40_v1,
                                      process.HLT_Photon36_R9Id90_HE10_Iso40_v1,
                                      process.HLT_Dimuon0_Jpsi_Muon_v1,
                                      process.HLT_Dimuon0_Upsilon_Muon_v1,
                                      process.HLT_Mu17_Mu8_SameSign_v1,
                                      process.HLT_Mu17_Mu8_SameSign_DPhi_v1,
                                      process.HLT_HT750_DisplacedDijet80_Inclusive_v1,
                                      process.HLT_HT650_DisplacedDijet80_Inclusive_v1,
                                      process.HLT_HT350_DisplacedDijet80_Tight_DisplacedTrack_v1,
                                      process.HLT_HT350_DisplacedDijet80_DisplacedTrack_v1,
                                      process.HLT_DoubleEle8_CaloIdL_TrkIdVL_Mass8_PFHT300_v1,
                                      process.HLT_Mu10_CentralPFJet40_BTagCSV_v1,
                                      process.HLT_Ele10_CaloId_TrackIdVL_CentralPFJet40_BTagCSV_v1,
                                      process.HLT_Ele15_IsoVVVL_BTagtop8CSV07_PFHT400_v1,
                                      process.HLT_Ele15_IsoVVVL_PFHT400_PFMET70_v1,
                                      process.HLT_Ele15_IsoVVVL_PFHT600_v1,
                                      process.HLT_Ele15_PFHT300_v1,
                                      process.HLT_Physics_v1,
                                      process.HLT_ReducedIterativeTracking_v1,
                                      process.HLT_ZeroBias_v1,
                                      process.HLT_IsoTrackHE_v16,
                                      process.HLT_IsoTrackHB_v15,
                                      process.HLTriggerFinalPath)
                                     )

# remove any instance of the FastTimerService
if 'FastTimerService' in process.__dict__:
    del process.FastTimerService

# instrument the menu with the FastTimerService
process.load( "HLTrigger.Timer.FastTimerService_cfi" )
process.load ("HLTrigger.Timer.fastTimerServiceClient_cfi" )

# this is currently ignored in 7.x, and alway uses the real tim clock
process.FastTimerService.useRealTimeClock         = False

# enable specific features
process.FastTimerService.enableTimingPaths        = True
process.FastTimerService.enableTimingModules      = True
process.FastTimerService.enableTimingExclusive    = True

# print a text summary at the end of the job
process.FastTimerService.enableTimingSummary      = True

# skip the first path (useful for HLT timing studies to disregard the time spent loading event and conditions data)
process.FastTimerService.skipFirstPath            = False

# enable per-event DQM plots
process.FastTimerService.enableDQM                = True

# enable per-path DQM plots
process.FastTimerService.enableDQMbyPathActive    = True
process.FastTimerService.enableDQMbyPathTotal     = True
process.FastTimerService.enableDQMbyPathOverhead  = True
process.FastTimerService.enableDQMbyPathDetails   = True
process.FastTimerService.enableDQMbyPathCounters  = True
process.FastTimerService.enableDQMbyPathExclusive = True

# enable per-module DQM plots
process.FastTimerService.enableDQMbyModule        = True
#process.FastTimerService.enableDQMbyModuleType    = True

# enable per-event DQM sumary plots
process.FastTimerService.enableDQMSummary         = True

# enable per-event DQM plots by lumisection
process.FastTimerService.enableDQMbyLumiSection   = True
process.FastTimerService.dqmLumiSectionsRange     = 2500    # lumisections (23.31 s)

# set the time resolution of the DQM plots
process.FastTimerService.dqmTimeRange             = 100000.   # ms
process.FastTimerService.dqmTimeResolution        =    5.   # ms
process.FastTimerService.dqmPathTimeRange         =  10000.   # ms
process.FastTimerService.dqmPathTimeResolution    =    0.5  # ms
process.FastTimerService.dqmModuleTimeRange       =   4000.   # ms
process.FastTimerService.dqmModuleTimeResolution  =    0.2  # ms

# set the base DQM folder for the plots
process.FastTimerService.dqmPath                  = "HLTNew1/TimerService"
process.fastTimerServiceClient.dqmPath            = "HLTNew1/TimerService"
process.FastTimerService.enableDQMbyProcesses     = True

# save the DQM plots in the DQMIO format
process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
                                     fileName = cms.untracked.string("DQMNew1.root"),
                                     outputCommands = cms.untracked.vstring(["drop CastorDataFramesSorted_mix_*_HLTNew1"])
)
process.FastTimerOutput = cms.EndPath( process.dqmOutput )

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)

process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.eventinterpretaion_step = cms.Path(process.EIsequence)
process.prevalidation_step = cms.Path(process.prevalidation)
process.dqmoffline_step = cms.Path(process.DQMOffline)
process.validation_step = cms.EndPath(process.validation)
process.endjob_step = cms.EndPath(process.endOfProcess)

process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step)
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.eventinterpretaion_step])
process.schedule.extend([process.analyze])
process.schedule.extend([process.FastTimerOutput])

# customisation of the process.
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1
process = customisePostLS1(process)

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC 

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforMC(process)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
#from SLHCUpgradeSimulations.Configuration.postLS1Customs import *
#process = customise_HLT(process)
#process = customisePostLS1(process)

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
#from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
#process = setCrossingFrameOn(process)

# End of customisation functions

