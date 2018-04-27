import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST2')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')

process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('hiRecoDM nevts:2'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    #'file:/data/abaty/VirginRaw_CentralitySkims/VirginRAW_2010_HICorePhysics_SKIM_Cent_0_5_1.root',
    #'file:/data/abaty/HLT_Emulated_2010Data/outputHIPhysicsVirginRaw.root'
    #'file:/data/abaty/VirginRaw_CentralitySkims/VirginRAW_2010_HICorePhysics_SKIM_Cent_0_5_10.root',
    #'/store/hidata/HIRun2015/HITrackerVirginRaw/RAW/v1/000/263/400/00000/40322926-4AA3-E511-95F7-02163E0146A8.root'
    #'file:/data/TrackerStudies/2015_VR_forBaselineFollowerStudies_Aug2017.root'
    'file:/data/TrackerStudies/RECORealistic100ev.root'
    #'file:ROOTFiles/RECORepackFromrawDataCollectro.root'
   )
)

#process.source = cms.Source("NewEventStreamFileReader",
#fileNames = cms.untracked.vstring('root://eoscms.cern.ch//store/t0streamer/Data/HIPhysicsVirginRaw/000/262/296/run262296_ls0223_streamHIPhysicsVirginRaw_StorageManager.dat'
#)
#)

# Output definition
process.RECOoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    fileName = cms.untracked.string('RECO.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RECO')
    ),
    # outputCommands = cms.untracked.vstring('keep *_siStripZeroSuppression_*_*')
    
)



process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = '75X_dataRun2_HLTHI_v4'

## Offline Silicon Tracker Zero Suppression
from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *
process.siStripZeroSuppression.produceRawDigis = cms.bool(False)
process.siStripZeroSuppression.produceCalculatedBaseline = cms.bool(False)
process.siStripZeroSuppression.produceBaselinePoints = cms.bool(False)
process.siStripZeroSuppression.storeCM = cms.bool(True)
process.siStripZeroSuppression.produceHybridFormat = cms.bool(False)
process.siStripZeroSuppression.Algorithms.CommonModeNoiseSubtractionMode = cms.string('Median')
process.siStripZeroSuppression.Algorithms.MeanCM = cms.int32(512)
process.siStripZeroSuppression.Algorithms.DeltaCMThreshold = cms.uint32(20)
process.siStripZeroSuppression.Algorithms.Use10bitsTruncation = cms.bool(True)   

process.siStripDigis.ProductLabel=cms.InputTag('myRawDataCollector')

process.TFileService = cms.Service("TFileService",
        fileName=cms.string("Baselines.root"))

process.hybridAna = cms.EDAnalyzer("SiStripHybridFormatAnalyzer",

    srcDigis =  cms.InputTag('siStripZeroSuppression','VirginRaw'),
    srcAPVCM =  cms.InputTag('siStripZeroSuppression','APVCMVirginRaw'),
    nModuletoDisplay = cms.uint32(10000),
    plotAPVCM	= cms.bool(True)
)


# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.siStripDigis)
#process.reconstruction_step = cms.Path(process.striptrackerlocalreco+process.moddedZS+process.moddedClust+process.baselineAna+process.moddedbaselineAna+process.clusterMatching)
process.reconstruction_step = cms.Path(process.siStripZeroSuppression)
#process.reconstruction_step = cms.Path(process.striptrackerlocalreco)
#process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOoutput_step = cms.EndPath(process.RECOoutput)


# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step, process.RECOoutput_step)
