# Auto generated configuration file
# using: 
# Revision: 1.222.2.6 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: hiRecoDM -n 10 --scenario HeavyIons -s RAW2DIGI,L1Reco,RECO --processName TEST --datatier GEN-SIM-RECO --eventcontent FEVTDEBUG --customise SimGeneral.DataMixingModule.DataMixer_DataConditions_3_8_X_data2010 --cust_function customise --geometry DB --filein file:DMRawSimOnReco_DIGI2RAW.root --fileout hiRecoDM_RECO.root --conditions FrontierConditions_GlobalTag,MC_38Y_V12::All --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')

process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('hiRecoDM nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.Timing = cms.Service("Timing")
#process.MessageLogger.cerr.threshold = cms.untracked.string("DEBUG")
#process.MessageLogger.categories+=cms.vstring("SiStripFedCMExtractor","SiStripProcessedRawDigiSkimProducer")

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    #'/store/express/Run2010B/ExpressPhysics/FEVT/Express-v2/000/146/417/10F981FF-5EC6-DF11-9657-0030486733B4.root'
    #'file:../testGenSimOnReco/SingleZmumu_MatchVertexDM_DIGI2RAW.root'
    #'file:DMRawSimOnReco_DIGI2RAW.root'
    #'file:DMRawSimOnReco_DIGI2RAW.root'
	'/store/data/Run2010B/HeavyIonTest/RAW/v1/000/146/421/E6B24CF0-5EC6-DF11-B52D-00304879FC6C.root'
    )
)


#process.source = cms.Source("NewEventStreamFileReader",
#fileNames = cms.untracked.vstring('file:Data.00148514.0021.A.storageManager.01.0000.dat'
#)
#)


# Output definition
process.RECOoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    #outputCommands = process.RECOEventContent.outputCommands,
    fileName = cms.untracked.string('hiReco_E6B24CF0-5EC6-DF11-B52D-00304879FC6C_10ev.root'),
	#fileName = cms.untracked.string('test.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RECO')
    )
)

# Additional output definition
#process.RECOoutput.outputCommands.extend(['keep *_siStripProcessedRawDigisSkim_*_*',
#                                               'keep *_*_APVCM_*'])

# Other statements
process.GlobalTag.globaltag = 'GR_R_39X_V1::All'
#process.GlobalTag.globaltag = 'MC_39Y_V4::All'

## Offline Silicon Tracker Zero Suppression
process.siStripZeroSuppression.Algorithms.PedestalSubtractionFedMode = cms.bool(False)
process.siStripZeroSuppression.Algorithms.CommonModeNoiseSubtractionMode = cms.string("IteratedMedian")
process.siStripZeroSuppression.doAPVRestore = cms.bool(True)
process.siStripZeroSuppression.produceRawDigis = cms.bool(True)
process.siStripZeroSuppression.produceCalculatedBaseline = cms.bool(True)
process.siStripZeroSuppression.storeCM = cms.bool(True)
process.siStripZeroSuppression.storeInZScollBadAPV = cms.bool(True)


    



process.TimerService = cms.Service("TimerService", useCPUtime = cms.untracked.bool(True) # set to false for wall-clock-time
)								  
								  
# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.siStripDigis)
process.reconstruction_step = cms.Path(process.trackerlocalreco)
#process.reconstruction_step = cms.Path(process.striptrackerlocalreco)
process.endjob_step = cms.Path(process.endOfProcess)
process.RECOoutput_step = cms.EndPath(process.RECOoutput)
#process.Timer_step = cms.Path(process.myTimer)

# Schedule definition
#process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.hipmonitor_step, process.RECOoutput_step)
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step, process.RECOoutput_step)
# customisation of the process






# End of customisation functions
