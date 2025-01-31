import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
## In line command options
options = VarParsing.VarParsing('analysis')
options.register('inputDataset',
                 '',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Input dataset")
options.register('runNumber',
                 379765,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run to process")
options.register('minFile',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Minimum file to process")
options.register('maxFile',
                 10,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Maximum file to process")
options.register('output',
                 './',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Output path")
options.parseArguments()

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '140X_dataRun3_Prompt_v4')

process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.debugModules = cms.untracked.vstring('DQMGenericClient')
# Enable LogInfo
process.MessageLogger.cerr = cms.untracked.PSet(
    # threshold = cms.untracked.string('ERROR'),
    WARNING = cms.untracked.PSet(
        limit = cms.untracked.int32(10000)
    ),
 )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

if options.inputDataset!='':
    listOfFiles = (os.popen("""dasgoclient -query="file dataset=%s instance=prod/global run=%i" """%(options.inputDataset, options.runNumber)).read()).split('\n')
    #listOfFiles = [os.path.abspath(file) for file in os.popen(f"dasgoclient -query=\"file dataset={options.inputDataset} instance=prod/global run={options.runNumber}\"").read().split('\n')]
#listOfFiles = listOfFiles[options.minFile:options.maxFile]
print(len(listOfFiles))
print(listOfFiles)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( listOfFiles[:-1] ),
    #fileNames = cms.untracked.vstring('root://xrootd-cms.infn.it//store/data/Run2024C/ScoutingPFMonitor/MINIAOD/PromptReco-v1/000/380/056/00000/dcf8ae98-736d-4376-bc6d-fc1dd2517304.root'),
    lumisToProcess = cms.untracked.VLuminosityBlockRange('%i:1-%i:max'%(options.runNumber, options.runNumber))
)

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
# turn on VID producer, indicate data format  to be
# DataFormat.AOD or DataFormat.MiniAOD, as appropriate 

#dataFormat = DataFormat.MiniAOD
#switchOnVIDElectronIdProducer(process, dataFormat)

# define which IDs we want to produce
#my_id_modules = ['RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Winter22_122X_V1_cff']

#add them to the VID producer
#for idmod in my_id_modules:
#    setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection)

#process.load("DQMOffline.Scouting.ScoutingMonitoring_cfi")
process.load("EventFilter.L1TRawToDigi.gtStage2Digis_cfi")
process.gtStage2Digis.InputLabel = cms.InputTag( "hltFEDSelectorL1" )
process.load("HLTriggerOffline.Scouting.ScoutingMuonTagProbeAnalyzer_cfi")
process.load("HLTriggerOffline.Scouting.ScoutingMuonTriggerAnalyzer_cfi")
process.load("HLTriggerOffline.Scouting.ScoutingMuonMonitoring_Client_cff")
#process.load("HLTriggerOffline.Scouting.MultiRunHarvester_cfg")
#process.load("DQMOffline.Scouting.PatMuonTagProbeAnalyzer_cfi")
process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMServices.FileIO.DQMFileSaverOnline_cfi")
process.dqmSaver.tag = '%i'%(options.minFile)
process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.uint32(1))
#process.dqmSaver.convention = 'Offline'
#process.content = cms.EDAnalyzer("EventContentAnalyzer")
#process.allPath = cms.Path(process.scoutingMonitoringTagProbeMuon * process.scoutingMonitoringTriggerMuon * process.muonEfficiency)

process.allPath = cms.Path(process.scoutingMonitoringTagProbeMuonNoVtx
                           * process.scoutingMonitoringTagProbeMuonVtx
                           * process.muonEfficiencyNoVtx
                           * process.muonEfficiencyVtx 
                           * process.scoutingMonitoringTriggerMuon
                           * process.muonTriggerEfficiency)
process.p = cms.EndPath(process.dqmSaver)
