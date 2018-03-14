import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST1')

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
    input = cms.untracked.int32(20)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    #'file:/data/abaty/VirginRaw_CentralitySkims/VirginRAW_2010_HICorePhysics_SKIM_Cent_0_5_1.root',
    #'file:/data/abaty/HLT_Emulated_2010Data/outputHIPhysicsVirginRaw.root'
    #'file:/data/abaty/VirginRaw_CentralitySkims/VirginRAW_2010_HICorePhysics_SKIM_Cent_0_5_10.root',
    #'/store/hidata/HIRun2015/HITrackerVirginRaw/RAW/v1/000/263/400/00000/40322926-4AA3-E511-95F7-02163E0146A8.root'
    'file:inputVR.root'
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
    )
)


process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = '75X_dataRun2_HLTHI_v4'

## Offline Silicon Tracker Zero Suppression
from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *
process.siStripZeroSuppression.Algorithms.PedestalSubtractionFedMode = cms.bool(False)
process.siStripZeroSuppression.Algorithms.CommonModeNoiseSubtractionMode = cms.string("IteratedMedian")
#process.siStripZeroSuppression.Algorithms.APVInspectMode = cms.string("Null")
process.siStripZeroSuppression.doAPVRestore = cms.bool(True)
process.siStripZeroSuppression.produceRawDigis = cms.bool(True)
process.siStripZeroSuppression.produceCalculatedBaseline = cms.bool(True)
process.siStripZeroSuppression.produceBaselinePoints = cms.bool(True)
process.siStripZeroSuppression.storeCM = cms.bool(True)
process.siStripZeroSuppression.storeInZScollBadAPV = cms.bool(True)


process.TFileService = cms.Service("TFileService",
        fileName=cms.string("Baselines.root"))

process.baselineAna = cms.EDAnalyzer("SiStripBaselineAnalyzer",
        Algorithms = DefaultAlgorithms,
        srcBaseline =  cms.InputTag('siStripZeroSuppression','BADAPVBASELINEVirginRaw'),
        srcBaselinePoints =  cms.InputTag('siStripZeroSuppression','BADAPVBASELINEPOINTSVirginRaw'),
        srcAPVCM  =  cms.InputTag('siStripZeroSuppression','APVCMVirginRaw'),
        #srcProcessedRawDigi =  cms.InputTag('siStripZeroSuppression','VirginRaw'),
        srcProcessedRawDigi =  cms.InputTag('siStripDigis','VirginRaw'),
        srcClusters =  cms.InputTag('siStripClusters',''),
        nModuletoDisplay = cms.uint32(1000),
        plotClusters = cms.bool(True),
        plotBaseline = cms.bool(True),
        plotBaselinePoints = cms.bool(True),
        plotRawDigi     = cms.bool(True),
        plotAPVCM   = cms.bool(True),
        plotPedestals = cms.bool(False)
)

process.moddedZS = process.siStripZeroSuppression.clone()
process.moddedZS.Algorithms.APVInspectMode = cms.string("BaselineFollower")
process.moddedZS.Algorithms.APVRestoreMode = cms.string("BaselineFollower")
process.moddedZS.Algorithms.consecThreshold = cms.uint32(9)#just for testing

process.moddedClust = process.siStripClusters.clone()
process.moddedClust.DigiProducersList = cms.VInputTag(
    cms.InputTag('siStripDigis','ZeroSuppressed'),
    cms.InputTag('moddedZS','VirginRaw'),
    cms.InputTag('moddedZS','ProcessedRaw'),
    cms.InputTag('moddedZS','ScopeMode'))

process.moddedbaselineAna = process.baselineAna.clone()
process.moddedbaselineAna.srcBaseline = cms.InputTag('moddedZS','BADAPVBASELINEVirginRaw')
process.moddedbaselineAna.srcBaselinePoints = cms.InputTag('moddedZS','BADAPVBASELINEPOINTSVirginRaw')
process.moddedbaselineAna.srcAPVCM = cms.InputTag('moddedZS','APVCMVirginRaw')
process.moddedbaselineAna.srcProcessedRawDigi = cms.InputTag('moddedZS','VirginRaw')
process.moddedbaselineAna.srcClusters = cms.InputTag('moddedClust','')

from RecoLocalTracker.SiStripZeroSuppression.SiStripBaselineComparitor_cfi import *
process.clusterMatching = cms.EDAnalyzer("SiStripBaselineComparitor",
    srcClusters =  cms.InputTag('siStripClusters',''),
    srcClusters2 =  cms.InputTag('moddedClust','')
)
								  
# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.siStripDigis)
process.reconstruction_step = cms.Path(process.striptrackerlocalreco+process.moddedZS+process.moddedClust+process.baselineAna+process.moddedbaselineAna+process.clusterMatching)
#process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOoutput_step = cms.EndPath(process.RECOoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step, process.RECOoutput_step)