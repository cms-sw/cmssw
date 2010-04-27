import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorModule.dtChamberEfficiency_Cosmics_cfi import *
from DQM.DTMonitorModule.dtSegmentTask_cfi import *
dtSegmentAnalysisMonitor.detailedAnalysis = True
dtSegmentAnalysisMonitor.slideTimeBins = False
dtSegmentAnalysisMonitor.nLSTimeBin = 5

from DQM.DTMonitorModule.dtResolutionTask_cfi import *

dqmInfoDT = cms.EDFilter("DQMEventInfo",
                         subSystemFolder = cms.untracked.string('DT')
                         )


dtDataIntegrityUnpacker = cms.EDProducer("DTUnpackingModule",
    dataType = cms.string('DDU'),
    inputLabel = cms.InputTag('source'),
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
DTDataIntegrityTask.offlineMode = True

from DQM.DTMonitorModule.dtTriggerEfficiencyTask_cfi import *

dtSourcesCosmics = cms.Sequence(dtDataIntegrityUnpacker  +
                                dtSegmentAnalysisMonitor +
                                dtResolutionAnalysisMonitor +
                                dtEfficiencyMonitor +
                                dtTriggerEfficiencyMonitor +
                                dqmInfoDT)
