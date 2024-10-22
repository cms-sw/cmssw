import FWCore.ParameterSet.Config as cms

### DQM Services ###

## DQM store ##
from DQMServices.Core.DQM_cfg import *
DQM.collectorHost = ''

## DQM file saving and environment ##
from DQMServices.Components.DQMEnvironment_cfi import *
# File saving #
dqmSaver.convention        = 'Online'
dqmSaver.dirName           = '.'
dqmSaver.producer          = 'DQM'
dqmSaver.saveByRun         = 1
dqmSaver.saveAtJobEnd      = True
dqmSaver.referenceHandling = 'qtests'
# Environment #
dqmEnv.subSystemFolder = 'SiStrip'

## ME2EDM/EDM2ME ##
from DQMServices.Components.MEtoEDMConverter_cfi import *
from DQMServices.Components.EDMtoMEConverter_cfi import *

## Quality tester ##
from DQMServices.Core.DQMQualityTester import DQMQualityTester
qTester = DQMQualityTester(
    qtList                  = cms.untracked.FileInPath( 'DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml' ),
    getQualityTestsFromFile = cms.untracked.bool( True ),
    prescaleFactor          = cms.untracked.int32( 1 )
)

### SiStrip DQM Modules ###

## SiStrip DQM reconstruction ##
from DQM.SiStripMonitorClient.SiStripDQMRecoConfigOfflineGlobalRunCAF_cfi   import *
## SiStrip DQM sources ##
from DQM.SiStripMonitorClient.SiStripDQMSourceConfigOfflineGlobalRunCAF_cfi import *
## SiStrip DQM client ##
from DQM.SiStripMonitorClient.SiStripClientConfig_cff import *
