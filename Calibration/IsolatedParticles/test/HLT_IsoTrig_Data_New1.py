# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:run2_mc_GRun -n 10 --eventcontent FEVTDEBUGHLT,DQM -s RAW2DIGI,L1Reco,RECO,EI,VALIDATION,DQM --datatier GEN-SIM-RECO,DQM --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --geometry Extended2015 --magField 38T_PostLS1 --conditions auto:run2_mc_GRun --no_exec --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms

process = cms.Process('HLTNew1')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.hlt_stage1_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('CommonTools.ParticleFlow.EITopPAG_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
        'file:9A19F942-E380-E111-8AEE-001D09F24303.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition

#process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
#    splitLevel = cms.untracked.int32(0),
#    eventAutoFlushCompressedSize = cms.untracked.int32(1048576),
#    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
#    fileName = cms.untracked.string('file:step3.root'),
#    dataset = cms.untracked.PSet(
#        filterName = cms.untracked.string(''),
#        dataTier = cms.untracked.string('GEN-SIM-RECO')
#    )
#)

#process.DQMoutput = cms.OutputModule("PoolOutputModule",
#    splitLevel = cms.untracked.int32(0),
#    outputCommands = process.DQMEventContent.outputCommands,
#    fileName = cms.untracked.string('file:step3_inDQM.root'),
#    dataset = cms.untracked.PSet(
#        filterName = cms.untracked.string(''),
#        dataTier = cms.untracked.string('DQM')
#    )
#)

process.TFileService = cms.Service("TFileService",
#                                   fileName = cms.string('/uscmst1b_scratch/lpc1/3DayLifetime/guptar/Try_HLTIsoTrig_Modified.root')
                                   fileName = cms.string('Try_HLTIsoTrig_New1.root')
                                   )

process.load('Calibration.IsolatedParticles.isoTrig_cfi')
process.IsoTrigHB.Verbosity = 0
#process.IsoTrigHB.Verbosity = 201
process.IsoTrigHB.ProcessName = "HLTNew1"
process.IsoTrigHB.DoL2L3 = True
process.IsoTrigHB.DoTimingTree = True
process.IsoTrigHB.DoStudyIsol = True
process.IsoTrigHB.DoChgIsolTree = True
process.IsoTrigHB.DoMipCutTree = False

process.IsoTrigHE = process.IsoTrigHB.clone(
    Triggers       = ["HLT_IsoTrackHE"],
    DoTimingTree = False)

process.analyze = cms.EndPath(process.IsoTrigHB + process.IsoTrigHE)

# Additional output definition

# Other statements
#process.mix.playback = True
#process.mix.digitizers = cms.PSet()
#for a in process.aliases: delattr(process, a)
#process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data_GRun', '')

process.load('Calibration.IsolatedParticles.HLT_IsoTrack_cff')
process.load('Calibration.HcalIsolatedTrackReco.isolEcalPixelTrackProd_cfi')
process.load('HLTrigger.special.ecalIsolPixelTrackFilter_cfi')

process.isolEcalPixelTrackProdHB = process.isolEcalPixelTrackProd.clone(
    filterLabel          = cms.InputTag("hltIsolPixelTrackL2FilterHB"),
    EBRecHitSource = cms.InputTag('hltEcalRecHit','EcalRecHitsEB'),
    EERecHitSource = cms.InputTag('hltEcalRecHit','EcalRecHitsEE')
    )

process.isolEcalPixelTrackProdHE = process.isolEcalPixelTrackProdHB.clone(
    filterLabel          = cms.InputTag("hltIsolPixelTrackL2FilterHE"),
    )
process.ecalIsolPixelTrackFilterHB = process.ecalIsolPixelTrackFilter.clone(
    candTag = cms.InputTag("isolEcalPixelTrackProdHB"),
    MaxEnergyIn = cms.double( 1.2 ),
    MaxEnergyOut = cms.double( 1.2 )
    )
process.ecalIsolPixelTrackFilterHE = process.ecalIsolPixelTrackFilter.clone(
    candTag = cms.InputTag("isolEcalPixelTrackProdHE"),
    MaxEnergyIn = cms.double( 1.2 ),
    MaxEnergyOut = cms.double( 1.2 )
    )
#process.hltIsolPixelTrackProdHE.maxPTrackForIsolation = 3.0
#process.hltIsolPixelTrackProdHB.maxPTrackForIsolation = 3.0
process.hltIsolPixelTrackProdHE.minPTrack = 10.0
process.hltIsolPixelTrackProdHB.minPTrack = 10.0

process.hltHITIPTCorrectorHE.filterLabel = "ecalIsolPixelTrackFilterHE"
process.hltHITIPTCorrectorHB.filterLabel = "ecalIsolPixelTrackFilterHB"


#process.myHLTSchedule = cms.Schedule( *(process.HLTriggerFirstPath,
#                                        process.HLT_IsoTrackHE_v16,
#                                        process.HLT_IsoTrackHB_v15,
#                                        process.HLTriggerFinalPath
#                                        ))

process.HLT_IsoTrackHE_v16 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet68 + process.hltPreIsoTrackHE + process.HLTDoLocalPixelSequence + process.hltPixelLayerTripletsHITHB + process.hltHITPixelTracksHB + process.hltPixelLayerTripletsHITHE + process.hltHITPixelTracksHE + process.hltHITPixelVerticesHE + process.hltIsolPixelTrackProdHE + process.hltIsolPixelTrackL2FilterHE + 
process.HLTDoFullUnpackingEgammaEcalSequence + process.isolEcalPixelTrackProdHE + process.ecalIsolPixelTrackFilterHE +
process.HLTDoLocalStripSequence + process.hltPixelLayerTriplets + process.hltHITPixelTripletSeedGeneratorHE + process.hltHITCkfTrackCandidatesHE + process.hltHITCtfWithMaterialTracksHE + process.hltHITIPTCorrectorHE + process.hltIsolPixelTrackL3FilterHE + process.HLTEndSequence )
process.HLT_IsoTrackHB_v15 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet68 + process.hltPreIsoTrackHB + process.HLTDoLocalPixelSequence + process.hltPixelLayerTripletsHITHB + process.hltHITPixelTracksHB + process.hltHITPixelVerticesHB + process.hltIsolPixelTrackProdHB + process.hltIsolPixelTrackL2FilterHB + 
process.HLTDoFullUnpackingEgammaEcalSequence + process.isolEcalPixelTrackProdHB + process.ecalIsolPixelTrackFilterHB +
process.HLTDoLocalStripSequence + process.hltPixelLayerTriplets + process.hltHITPixelTripletSeedGeneratorHB + process.hltHITCkfTrackCandidatesHB + process.hltHITCtfWithMaterialTracksHB + process.hltHITIPTCorrectorHB + process.hltIsolPixelTrackL3FilterHB + process.HLTEndSequence )

process.HLTSchedule = cms.Schedule( *(process.HLTAnalyzerEndpath, process.HLT_AK8PFJet360TrimMod_Mass30_v1, process.HLT_Dimuon13_PsiPrime_v1, process.HLT_Dimuon13_Upsilon_v1, process.HLT_Dimuon20_Jpsi_v1, process.HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_MW_v1, process.HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v1, process.HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v1, process.HLT_DoubleMu33NoFiltersNoVtx_v1, process.HLT_DoubleMu38NoFiltersNoVtx_v1, process.HLT_DoubleMu4_3_Bs_v1, process.HLT_DoubleMu4_3_Jpsi_Displaced_v1, process.HLT_DoubleMu4_JpsiTrk_Displaced_v1, process.HLT_DoubleMu4_LowMassNonResonantTrk_Displaced_v1, process.HLT_DoubleMu4_PsiPrimeTrk_Displaced_v1, process.HLT_DoublePhoton85_v1, process.HLT_Ele17_Ele12_Ele10_CaloId_TrackId_v1, process.HLT_Ele20WP60_Ele8_Mass55_v1, process.HLT_Ele22_eta2p1_WP85_Gsf_LooseIsoPFTau20_v1, process.HLT_Ele23_Ele12_CaloId_TrackId_Iso_v1, process.HLT_Ele25WP60_SC4_Mass55_v1, process.HLT_Ele27_eta2p1_WP85_Gsf_CentralPFJet30_BTagCSV_v1, process.HLT_Ele27_eta2p1_WP85_Gsf_TriCentralPFJet40_v1, process.HLT_Ele27_eta2p1_WP85_Gsf_TriCentralPFJet60_50_35_v1, process.HLT_Ele27_eta2p1_WP85_Gsf_v1, process.HLT_Ele32_eta2p1_WP85_Gsf_CentralPFJet30_BTagCSV_v1, process.HLT_Ele32_eta2p1_WP85_Gsf_TriCentralPFJet40_v1, process.HLT_Ele32_eta2p1_WP85_Gsf_TriCentralPFJet60_50_35_v1, process.HLT_Ele32_eta2p1_WP85_Gsf_v1, process.HLT_Ele45_CaloIdVT_GsfTrkIdT_PFJet200_PFJet50_v1, process.HLT_Ele95_CaloIdVT_GsfTrkIdT_v1, process.HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v1, process.HLT_IsoMu20_eta2p1_IterTrk02_CentralPFJet30_BTagCSV_v1, process.HLT_IsoMu20_eta2p1_IterTrk02_TriCentralPFJet40_v1, process.HLT_IsoMu20_eta2p1_IterTrk02_TriCentralPFJet60_50_35_v1, process.HLT_IsoMu20_eta2p1_IterTrk02_v1, process.HLT_IsoMu24_IterTrk02_v1, process.HLT_IsoMu24_eta2p1_IterTrk02_CentralPFJet30_BTagCSV_v1, process.HLT_IsoMu24_eta2p1_IterTrk02_TriCentralPFJet40_v1, process.HLT_IsoMu24_eta2p1_IterTrk02_TriCentralPFJet60_50_35_v1, process.HLT_IsoMu24_eta2p1_IterTrk02_v1, 
#HLT_IsoTkMu20_eta2p1_IterTrk02_v1, 
#HLT_IsoTkMu24_IterTrk02_v1, 
#HLT_IsoTkMu24_eta2p1_IterTrk02_v1, 
process.HLT_JetE30_NoBPTX3BX_NoHalo_v1, process.HLT_JetE30_NoBPTX_v1, process.HLT_JetE50_NoBPTX3BX_NoHalo_v1, process.HLT_JetE70_NoBPTX3BX_NoHalo_v1, process.HLT_L2DoubleMu23_NoVertex_v1, process.HLT_L2DoubleMu28_NoVertex_2Cha_Angle2p5_Mass10_v1, process.HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_Mass10_v1, process.HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v1, process.HLT_L2Mu10_NoVertex_NoBPTX_v1, process.HLT_L2Mu20_NoVertex_3Sta_NoBPTX3BX_NoHalo_v1, process.HLT_L2Mu30_NoVertex_3Sta_NoBPTX3BX_NoHalo_v1, process.HLT_LooseIsoPFTau50_Trk30_eta2p1_MET120_v1, process.HLT_Mu17_Mu8_v1, process.HLT_Mu17_TkMu8_v1, process.HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v1, process.HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v1, process.HLT_Mu23_TrkIsoVVL_Ele12_Gsf_CaloId_TrackId_Iso_MediumWP_v1, process.HLT_Mu25_TkMu0_dEta18_Onia_v1, process.HLT_Mu30_TkMu11_v1, process.HLT_Mu38NoFiltersNoVtx_Photon38_CaloIdL_v1, process.HLT_Mu40_eta2p1_PFJet200_PFJet50_v1, process.HLT_Mu40_v1, process.HLT_Mu42NoFiltersNoVtx_Photon42_CaloIdL_v1, process.HLT_Mu8_TrkIsoVVL_Ele23_Gsf_CaloId_TrackId_Iso_MediumWP_v1, process.HLT_PFHT350_PFMET120_NoiseCleaned_v1, process.HLT_PFHT900_v1, process.HLT_PFJet260_v1, process.HLT_PFMET120_NoiseCleaned_BTagCSV07_v1, process.HLT_PFMET170_NoiseCleaned_v1, process.HLT_Photon135_PFMET40_v1, process.HLT_Photon135_VBF_v1, process.HLT_Photon150_PFMET40_v1, process.HLT_Photon150_VBF_v1, process.HLT_Photon155_v1, process.HLT_Photon160_PFMET40_v1, process.HLT_Photon160_VBF_v1, process.HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_PFMET40_v1, process.HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_VBF_v1, process.HLT_Photon250_NoHE_PFMET40_v1, process.HLT_Photon250_NoHE_VBF_v1, process.HLT_Photon300_NoHE_PFMET40_v1, process.HLT_Photon300_NoHE_VBF_v1, process.HLT_Photon36_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon18_AND_HE10_R9Id65_Mass95_v1, process.HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_PFMET40_v1, process.HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_VBF_v1, process.HLT_Photon42_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon22_AND_HE10_R9Id65_v1, process.HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_PFMET40_v1, process.HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_VBF_v1, process.HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_PFMET40_v1, process.HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_VBF_v1, process.HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_PFMET40_v1, process.HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_VBF_v1, process.HLT_Physics_v1, process.HLT_ReducedIterativeTracking_v1, process.HLT_ZeroBias_v1, process.HLT_IsoTrackHE_v16, process.HLT_IsoTrackHB_v15, process.HLTriggerFinalPath, process.HLTriggerFirstPath ))

## remove any instance of the FastTimerService
#if 'FastTimerService' in process.__dict__:
#    del process.FastTimerService
#
## instrument the menu with the FastTimerService
#process.load( "HLTrigger.Timer.FastTimerService_cfi" )
#
## this is currently ignored in 7.x, and alway uses the real tim clock
#process.FastTimerService.useRealTimeClock         = True
#
## enable specific features
#process.FastTimerService.enableTimingPaths        = True
#process.FastTimerService.enableTimingModules      = True
#process.FastTimerService.enableTimingExclusive    = True
#
## print a text summary at the end of the job
#process.FastTimerService.enableTimingSummary      = True
#
## skip the first path (useful for HLT timing studies to disregard the time spent loading event and conditions data)
#process.FastTimerService.skipFirstPath            = False
#
## enable per-event DQM plots
#process.FastTimerService.enableDQM                = True
#
## enable per-path DQM plots
#process.FastTimerService.enableDQMbyPathActive    = True
#process.FastTimerService.enableDQMbyPathTotal     = True
#process.FastTimerService.enableDQMbyPathOverhead  = True
#process.FastTimerService.enableDQMbyPathDetails   = True
#process.FastTimerService.enableDQMbyPathCounters  = True
#process.FastTimerService.enableDQMbyPathExclusive = True
#
## enable per-module DQM plots
#process.FastTimerService.enableDQMbyModule        = True
#process.FastTimerService.enableDQMbyModuleType    = True
#
## enable per-event DQM sumary plots
#process.FastTimerService.enableDQMSummary         = True
#
## enable per-event DQM plots by lumisection
#process.FastTimerService.enableDQMbyLumiSection   = True
#process.FastTimerService.dqmLumiSectionsRange     = 2500    # lumisections (23.31 s)
#
## set the time resolution of the DQM plots
#process.FastTimerService.dqmTimeRange             = 100000.   # ms
#process.FastTimerService.dqmTimeResolution        =    5.   # ms
#process.FastTimerService.dqmPathTimeRange         =  10000.   # ms
#process.FastTimerService.dqmPathTimeResolution    =    0.5  # ms
#process.FastTimerService.dqmModuleTimeRange       =   4000.   # ms
#process.FastTimerService.dqmModuleTimeResolution  =    0.2  # ms
#
## set the base DQM folder for the plots
#process.FastTimerService.dqmPath                  = "HLTNew1/TimerService"
#process.FastTimerService.enableDQMbyProcesses     = True
#
## save the DQM plots in the DQMIO format
#process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
#    fileName = cms.untracked.string("DQMNew1.root"),
##    outputCommands = cms.untracked.vstring(["drop *_mix_*_HLTNew1"])
##CastorDataFramesSorted, mix, , HLTNew1
#)
#process.FastTimerOutput = cms.EndPath( process.dqmOutput )

# Path and EndPath definitions
#process.digitisation_step = cms.Path(process.pdigi)
#process.L1simulation_step = cms.Path(process.SimL1Emulator)
#process.digi2raw_step = cms.Path(process.DigiToRaw)

process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.eventinterpretaion_step = cms.Path(process.EIsequence)
process.prevalidation_step = cms.Path(process.prevalidation)
process.dqmoffline_step = cms.Path(process.DQMOffline)
process.validation_step = cms.EndPath(process.validation)
process.endjob_step = cms.EndPath(process.endOfProcess)
#process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
#process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
#process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.eventinterpretaion_step,process.prevalidation_step,process.validation_step,process.dqmoffline_step,process.endjob_step,process.FEVTDEBUGHLToutput_step,process.DQMoutput_step)
#process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step)
process.schedule = cms.Schedule(process.HLTSchedule)
#process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.eventinterpretaion_step])
process.schedule.extend([process.analyze])
#process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
process.schedule.extend([process.endjob_step])
#process.schedule.extend([process.FastTimerOutput])
# customisation of the process.
# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
#from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC 

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
#process = customizeHLTforMC(process)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import *
process = customise_HLT(process)
process = customisePostLS1(process)

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions

