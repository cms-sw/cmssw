import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorModule.dtChamberEfficiency_Cosmics_cfi import *
from DQM.DTMonitorModule.dtSegmentTask_cfi import *
from DQM.DTMonitorModule.dtDCSByLumiTask_cfi import *
from DQM.DTMonitorModule.dtRunConditionVar_cfi import *
dtSegmentAnalysisMonitor.detailedAnalysis = True
dtSegmentAnalysisMonitor.slideTimeBins = False
dtSegmentAnalysisMonitor.nLSTimeBin = 5

from DQM.DTMonitorModule.dtResolutionTask_cfi import *

dqmInfoDT = cms.EDAnalyzer("DQMEventInfo",
                         subSystemFolder = cms.untracked.string('DT')
                         )

# DT digitization and reconstruction
# Switched to TwinMux
from EventFilter.L1TXRawToDigi.twinMuxStage2Digis_cfi import *
twinMuxStage2Digis.DTTM7_FED_Source = 'rawDataCollector'

from EventFilter.DTRawToDigi.dtunpackerDDUGlobal_cfi import *
#from EventFilter.DTRawToDigi.dtunpackerDDULocal_cfi import *
dtunpacker.readOutParameters.performDataIntegrityMonitor = True
dtunpacker.readOutParameters.rosParameters.performDataIntegrityMonitor = True
dtunpacker.readOutParameters.debug = False
dtunpacker.readOutParameters.rosParameters.debug = False
dtunpacker.inputLabel = 'rawDataCollector'

unpackers = cms.Sequence(dtunpacker + twinMuxStage2Digis )


dtDataIntegrityUnpacker = cms.EDProducer("DTUnpackingModule",
    dataType = cms.string('DDU'),
    inputLabel = cms.InputTag('rawDataCollector'),
    fedbyType = cms.bool(False),
    useStandardFEDid = cms.bool(True),
    dqmOnly = cms.bool(True),                       
    readOutParameters = cms.PSet(
        debug = cms.untracked.bool(False),
        rosParameters = cms.PSet(
            writeSC = cms.untracked.bool(True),
            readingDDU = cms.untracked.bool(True),
            performDataIntegrityMonitor = cms.untracked.bool(True),
            readDDUIDfromDDU = cms.untracked.bool(True),
            debug = cms.untracked.bool(False),
            localDAQ = cms.untracked.bool(False)
        ),
        localDAQ = cms.untracked.bool(False),
        performDataIntegrityMonitor = cms.untracked.bool(True)
    )
)

from DQM.DTMonitorModule.dtDataIntegrityTask_cfi import *
DTDataIntegrityTask.processingMode = "Offline"

from DQM.DTMonitorModule.dtTriggerEfficiencyTask_cfi import *

dtSourcesCosmics = cms.Sequence(#dtDataIntegrityUnpacker  +
			 unpackers  +
                         DTDataIntegrityTask +
                         dtDCSByLumiMonitor + 
                         dtRunConditionVar + 
                         dtSegmentAnalysisMonitor +
                         dtResolutionAnalysisMonitor +
                         dtEfficiencyMonitor +
                         dtTriggerEfficiencyMonitor +
                         dqmInfoDT)
