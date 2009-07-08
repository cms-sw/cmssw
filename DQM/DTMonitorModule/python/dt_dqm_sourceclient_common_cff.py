import FWCore.ParameterSet.Config as cms



# filter on trigger type
calibrationEventsFilter = cms.EDFilter("HLTTriggerTypeFilter",
                                       # 1=Physics, 2=Calibration, 3=Random, 4=Technical
                                       SelectedTriggerType = cms.int32(2) 
                                       )

# filter on trigger type
physicsEventsFilter = cms.EDFilter("HLTTriggerTypeFilter",
                                   # 1=Physics, 2=Calibration, 3=Random, 4=Technical
                                   SelectedTriggerType = cms.int32(1) 
                                   )




# DT digitization and reconstruction
from EventFilter.DTTFRawToDigi.dttfunpacker_cfi import *

from EventFilter.DTRawToDigi.dtunpackerDDUGlobal_cfi import *
#from EventFilter.DTRawToDigi.dtunpackerDDULocal_cfi import *
dtunpacker.readOutParameters.performDataIntegrityMonitor = True
dtunpacker.readOutParameters.rosParameters.performDataIntegrityMonitor = True
dtunpacker.readOutParameters.debug = False
dtunpacker.readOutParameters.rosParameters.debug = False


from RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff import *
dt1DRecHits.dtDigiLabel = 'dtunpacker'
#DTLinearDriftAlgo_CosmicData.recAlgoConfig.hitResolution = 0.05


from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *





# Data integrity
from DQM.DTMonitorModule.dtDataIntegrityTask_cfi import *
from DQM.DTMonitorClient.dtDataIntegrityTest_cfi import *
from DQM.DTMonitorClient.dtBlockedROChannelsTest_cfi import *


# Digi task
from DQM.DTMonitorModule.dtDigiTask_cfi import *
from DQM.DTMonitorClient.dtOccupancyTest_cfi import *
dtDigiMonitor.readDB = False 
dtDigiMonitor.filterSyncNoise = True
dtDigiMonitor.lookForSyncNoise = True

# Local Trigger task
from DQM.DTMonitorModule.dtTriggerTask_cfi import *
from DQM.DTMonitorClient.dtLocalTriggerTest_cfi import *
    
# segment reco task
from DQM.DTMonitorModule.dtSegmentTask_cfi import *
from DQM.DTMonitorClient.dtSegmentAnalysisTest_cfi import *

# resolution task
from DQM.DTMonitorModule.dtResolutionTask_cfi import *

# noise task
from DQM.DTMonitorModule.dtNoiseTask_cfi import *
from DQM.DTMonitorClient.dtNoiseAnalysis_cfi import *
dtNoiseAnalysisMonitor.doSynchNoise = True

# report summary
from DQM.DTMonitorClient.dtSummaryClients_cfi import *

dtqTester = cms.EDFilter("QualityTester",
                         #reportThreshold = cms.untracked.string('red'),
                         prescaleFactor = cms.untracked.int32(1),
                         qtList = cms.untracked.FileInPath('DQM/DTMonitorClient/test/QualityTests.xml'),
                         getQualityTestsFromFile = cms.untracked.bool(True)
                         )


# test pulse monitoring
from DQM.DTMonitorModule.dtDigiTask_TP_cfi import *
from DQM.DTMonitorClient.dtOccupancyTest_TP_cfi import *

# Local Trigger task for test pulses
from DQM.DTMonitorModule.dtTriggerTask_TP_cfi import *
from DQM.DTMonitorClient.dtLocalTriggerTest_TP_cfi import *


unpackers = cms.Sequence(dtunpacker + dttfunpacker)

reco = cms.Sequence(dt1DRecHits + dt4DSegments)

dtDQMTask = cms.Sequence(dtDigiMonitor + dtSegmentAnalysisMonitor + dtTriggerMonitor + dtNoiseMonitor + dtResolutionAnalysisMonitor)

dtDQMTest = cms.Sequence(dataIntegrityTest + blockedROChannelTest + triggerTest + dtOccupancyTest + segmentTest + dtNoiseAnalysisMonitor + dtSummaryClients + dtqTester)

dtDQMCalib = cms.Sequence(dtTPmonitor + dtTPTriggerMonitor + dtTPmonitorTest + dtTPTriggerTest)
