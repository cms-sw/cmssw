# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: --python_filename step3_RECO_Mustache_cfg.py --eventcontent RECOSIM --datatier GEN-SIM-RECO --fileout file:step3.root --conditions 123X_mcRun3_2021_realistic_v11 --step RAW2DIGI,L1Reco,RECO,RECOSIM,PAT --geometry DB:Extended --filein file:step2.root --era Run3,ctpps_2018 --no_exec --mc --procModifier ecal_deepsc
import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('standard')
options.register('inputFile',
                 'file:step2.root',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "intputFile")
options.register('outputFile',
                 'file:step3.root',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "outputFile")
                
options.parseArguments()


from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
from Configuration.ProcessModifiers.ecal_deepsc_cff import ecal_deepsc

process = cms.Process('ECALClustering',Run3,ctpps_2018,ecal_deepsc)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.PATMC_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(options.inputFile),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('--python_filename nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('step3.root'),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '125X_mcRun3_2022_realistic_v4', '')

# Setup the PFRechitThreshold configuration
process.myPFRechitThres = cms.ESSource("PoolDBESSource",
     connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
     toGet = cms.VPSet(
         cms.PSet(
             record = cms.string('EcalPFRecHitThresholdsRcd'),
             tag = cms.string('EcalPFRecHitThresholds_UL_2018_2e3sig')
         )
     )
)
process.es_prefer_pfRechitThres = cms.ESPrefer("PoolDBESSource","myPFRechitThres")

process.myTPGLinearization = cms.ESSource("PoolDBESSource",
     connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
     toGet = cms.VPSet(
         cms.PSet(
             record = cms.string('EcalTPGLinearizationConstRcd'),
             tag = cms.string('EcalTPGLinearizationConst_UL_2018_mc')
         )
     )
)
process.es_prefer_tpgLinearization = cms.ESPrefer("PoolDBESSource","myTPGLinearization")

process.TFileService = cms.Service("TFileService",
                                   #fileName = cms.string(options.outputFile)
                                   fileName = cms.string('output.root')
)


# DEBUG logging
# process.load("FWCore.MessageService.MessageLogger_cfi")
# process.MessageLogger.cerr.threshold = "DEBUG"
# process.MessageLogger.debugModules = ["particleFlowSuperClusterECAL"]



#### DeepSC customization
process.particleFlowSuperClusterECAL.deepSuperClusterConfig.modelFiles = \
    cms.vstring("RecoEcal/EgammaClusterProducers/data/DeepSCModels/ACAT2022/simpler_rechits/model_smallpadding.pb",
                "RecoEcal/EgammaClusterProducers/data/DeepSCModels/ACAT2022/simpler_rechits/model_largepadding.pb")

process.particleFlowSuperClusterECAL.deepSuperClusterConfig.configFileClusterFeatures = cms.string("RecoEcal/EgammaClusterProducers/data/DeepSCModels/ACAT2022/config_clusters_inputs.txt")
process.particleFlowSuperClusterECAL.deepSuperClusterConfig.configFileWindowFeatures = cms.string("RecoEcal/EgammaClusterProducers/data/DeepSCModels/ACAT2022/config_window_inputs.txt")
process.particleFlowSuperClusterECAL.deepSuperClusterConfig.configFileHitsFeatures = cms.string("RecoEcal/EgammaClusterProducers/data/DeepSCModels/ACAT2022/config_hits_inputs.txt")

process.particleFlowSuperClusterECAL.deepSuperClusterConfig.nClusterFeatures = cms.uint32(15)
process.particleFlowSuperClusterECAL.deepSuperClusterConfig.nWindowFeatures = cms.uint32(6)
process.particleFlowSuperClusterECAL.deepSuperClusterConfig.maxNClusters = cms.vuint32(15,30)
process.particleFlowSuperClusterECAL.deepSuperClusterConfig.maxNRechits = cms.vuint32(20,60)


# Same customization for OOTEcal producer
process.particleFlowSuperClusterOOTECAL.deepSuperClusterConfig.modelFiles = \
    cms.vstring("RecoEcal/EgammaClusterProducers/data/DeepSCModels/ACAT2022/simpler_rechits/model_smallpadding.pb",
                "RecoEcal/EgammaClusterProducers/data/DeepSCModels/ACAT2022/simpler_rechits/model_largepadding.pb")

process.particleFlowSuperClusterOOTECAL.deepSuperClusterConfig.configFileClusterFeatures = cms.string("RecoEcal/EgammaClusterProducers/data/DeepSCModels/ACAT2022/config_clusters_inputs.txt")
process.particleFlowSuperClusterOOTECAL.deepSuperClusterConfig.configFileWindowFeatures = cms.string("RecoEcal/EgammaClusterProducers/data/DeepSCModels/ACAT2022/config_window_inputs.txt")
process.particleFlowSuperClusterOOTECAL.deepSuperClusterConfig.configFileHitsFeatures = cms.string("RecoEcal/EgammaClusterProducers/data/DeepSCModels/ACAT2022/config_hits_inputs.txt")

process.particleFlowSuperClusterOOTECAL.deepSuperClusterConfig.nClusterFeatures = cms.uint32(15)
process.particleFlowSuperClusterOOTECAL.deepSuperClusterConfig.nWindowFeatures = cms.uint32(6)
process.particleFlowSuperClusterOOTECAL.deepSuperClusterConfig.maxNClusters = cms.vuint32(15,30)
process.particleFlowSuperClusterOOTECAL.deepSuperClusterConfig.maxNRechits = cms.vuint32(20,60)


# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.recosim_step = cms.Path(process.recosim)
process.Flag_BadChargedCandidateFilter = cms.Path(process.BadChargedCandidateFilter)
process.Flag_BadChargedCandidateSummer16Filter = cms.Path(process.BadChargedCandidateSummer16Filter)
process.Flag_BadPFMuonDzFilter = cms.Path(process.BadPFMuonDzFilter)
process.Flag_BadPFMuonFilter = cms.Path(process.BadPFMuonFilter)
process.Flag_BadPFMuonSummer16Filter = cms.Path(process.BadPFMuonSummer16Filter)
process.Flag_CSCTightHalo2015Filter = cms.Path(process.CSCTightHalo2015Filter)
process.Flag_CSCTightHaloFilter = cms.Path(process.CSCTightHaloFilter)
process.Flag_CSCTightHaloTrkMuUnvetoFilter = cms.Path(process.CSCTightHaloTrkMuUnvetoFilter)
process.Flag_EcalDeadCellBoundaryEnergyFilter = cms.Path(process.EcalDeadCellBoundaryEnergyFilter)
process.Flag_EcalDeadCellTriggerPrimitiveFilter = cms.Path(process.EcalDeadCellTriggerPrimitiveFilter)
process.Flag_HBHENoiseFilter = cms.Path(process.HBHENoiseFilterResultProducer+process.HBHENoiseFilter)
process.Flag_HBHENoiseIsoFilter = cms.Path(process.HBHENoiseFilterResultProducer+process.HBHENoiseIsoFilter)
process.Flag_HcalStripHaloFilter = cms.Path(process.HcalStripHaloFilter)
process.Flag_METFilters = cms.Path(process.metFilters)
process.Flag_chargedHadronTrackResolutionFilter = cms.Path(process.chargedHadronTrackResolutionFilter)
process.Flag_ecalBadCalibFilter = cms.Path(process.ecalBadCalibFilter)
process.Flag_ecalLaserCorrFilter = cms.Path(process.ecalLaserCorrFilter)
process.Flag_eeBadScFilter = cms.Path(process.eeBadScFilter)
process.Flag_globalSuperTightHalo2016Filter = cms.Path(process.globalSuperTightHalo2016Filter)
process.Flag_globalTightHalo2016Filter = cms.Path(process.globalTightHalo2016Filter)
process.Flag_goodVertices = cms.Path(process.primaryVertexFilter)
process.Flag_hcalLaserEventFilter = cms.Path(process.hcalLaserEventFilter)
process.Flag_hfNoisyHitsFilter = cms.Path(process.hfNoisyHitsFilter)
process.Flag_muonBadTrackFilter = cms.Path(process.muonBadTrackFilter)
process.Flag_trackingFailureFilter = cms.Path(process.goodVertices+process.trackingFailureFilter)
process.Flag_trkPOGFilters = cms.Path(process.trkPOGFilters)
process.Flag_trkPOG_logErrorTooManyClusters = cms.Path(~process.logErrorTooManyClusters)
process.Flag_trkPOG_manystripclus53X = cms.Path(~process.manystripclus53X)
process.Flag_trkPOG_toomanystripclus53X = cms.Path(~process.toomanystripclus53X)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)

# from RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi import *
# process.ecalDrivenGsfElectrons.EleDNNPFid= dict(
#         modelsFiles = [
#             "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Summer21_120X/lowpT/lowpT_modelDNN.pb",
#             "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Summer21_120X/EB_highpT/barrel_highpT_modelDNN.pb",
#             "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Summer21_120X/EE_highpT/endcap_highpT_modelDNN.pb"
#         ],
#         scalersFiles = [
#             "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Summer21_120X/lowpT/lowpT_scaler.txt",
#             "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Summer21_120X/EB_highpT/barrel_highpT_scaler.txt",
#             "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Summer21_120X/EE_highpT/endcap_highpT_scaler.txt"
#         ]
# )
# process.ecalDrivenGsfElectrons.EleDNNPFid.enabled = False

# from RecoEgamma.EgammaPhotonProducers.gedPhotons_cfi import *
# process.gedPhotons.PhotonDNNPFid = dict(
#         modelsFiles = [ 
#             "RecoEgamma/PhotonIdentification/data/Photon_PFID_dnn/Run3Summer21_120X/EB/barrel_modelDNN.pb",
#             "RecoEgamma/PhotonIdentification/data/Photon_PFID_dnn/Run3Summer21_120X/EE/endcap_modelDNN.pb"
#         ],
#         scalersFiles = [
#             "RecoEgamma/PhotonIdentification/data/Photon_PFID_dnn/Run3Summer21_120X/EB/barrel_scaler.txt",
#             "RecoEgamma/PhotonIdentification/data/Photon_PFID_dnn/Run3Summer21_120X/EE/endcap_scaler.txt" 
#         ]
# )
# process.gedPhotons.PhotonDNNPFid.enabled = False
# process.particleFlowTmp.PFEGammaFiltersParameters.useElePFidDnn = False
# process.particleFlowTmp.PFEGammaFiltersParameters.usePhotonPFidDnn = False

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.recosim_step,process.Flag_HBHENoiseFilter,process.Flag_HBHENoiseIsoFilter,process.Flag_CSCTightHaloFilter,process.Flag_CSCTightHaloTrkMuUnvetoFilter,process.Flag_CSCTightHalo2015Filter,process.Flag_globalTightHalo2016Filter,process.Flag_globalSuperTightHalo2016Filter,process.Flag_HcalStripHaloFilter,process.Flag_hcalLaserEventFilter,process.Flag_EcalDeadCellTriggerPrimitiveFilter,process.Flag_EcalDeadCellBoundaryEnergyFilter,process.Flag_ecalBadCalibFilter,process.Flag_goodVertices,process.Flag_eeBadScFilter,process.Flag_ecalLaserCorrFilter,process.Flag_trkPOGFilters,process.Flag_chargedHadronTrackResolutionFilter,process.Flag_muonBadTrackFilter,process.Flag_BadChargedCandidateFilter,process.Flag_BadPFMuonFilter,process.Flag_BadPFMuonDzFilter,process.Flag_hfNoisyHitsFilter,process.Flag_BadChargedCandidateSummer16Filter,process.Flag_BadPFMuonSummer16Filter,process.Flag_trkPOG_manystripclus53X,process.Flag_trkPOG_toomanystripclus53X,process.Flag_trkPOG_logErrorTooManyClusters,process.Flag_METFilter, process.endjob_step)
process.schedule.associate(process.patTask)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)


# customisation of the process.
# Automatic addition of the customisation function from Validation.Performance.IgProfInfo
from Validation.Performance.IgProfInfo import customise 
#call to customisation function customise imported from Validation.Performance.IgProfInfo
process = customise(process)



# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC 

#call to customisation function miniAOD_customizeAllMC imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
process = miniAOD_customizeAllMC(process)



# End of customisation functions

# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
