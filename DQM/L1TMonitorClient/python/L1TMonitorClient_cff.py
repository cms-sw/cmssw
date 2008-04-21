import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1THcalClient_cff import *
#include "DQM/L1TMonitorClient/data/L1TDTTFClient.cff"
from DQM.L1TMonitorClient.L1TDTTPGClient_cff import *
from DQM.L1TMonitorClient.L1TdeECALClient_cff import *
#    # use include file for dqmEnv dqmSaver
from DQMServices.Components.test.dqm_onlineEnv_cfi import *
#       # put your subsystem name here:
#       # DT, Ecal, Hcal, SiStrip, Pixel, RPC, CSC, L1T
#       # (this goes into the filename)
#       replace dqmSaver.fileName      = "L1T"
#       replace dqmSaver.dirName       = "."
#       # (this goes into the foldername)
#       replace dqmEnv.subSystemFolder = "Env"
# optionally change fileSaving  conditions
# replace dqmSaver.prescaleLS =   -1
# replace dqmSaver.prescaleTime = -1 # in minutes
# replace dqmSaver.prescaleEvt =  -1
# replace dqmSaver.saveAtRunEnd = true
# replace dqmSaver.saveAtJobEnd = false
# will add switch to select histograms to be saved soon
# run the quality tests as defined in QualityTests.xml
#    module qTester = QualityTester {
#	untracked int32 QualityTestPrescaler = 5000
#        untracked int32 prescaleFactor = 1
#        untracked bool getQualityTestsFromFile = true
#        untracked string qtList = "QualityTests.xml"
#        untracked string reportThreshold = "red" 
#    }
#    module qTester = QualityTester {
#      untracked string qtList = "QualityTests.xml"
#      untracked bool getQualityTestsFromFile = true
#      untracked int32 QualityTestPrescaler = 1
#    }
#
# END ################################################
#
l1tmonitorClient = cms.Path(l1thcalseqClient*l1tdttpgseqClient*l1tdeEcalseqClient*dqmEnv*dqmSaver)

