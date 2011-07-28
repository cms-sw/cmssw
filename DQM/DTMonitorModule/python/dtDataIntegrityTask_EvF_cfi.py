import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorModule.dtDataIntegrityTask_cfi import *
DTDataIntegrityTask.processingMode = "HLT"
DTDataIntegrityTask.fedIntegrityFolder = "DT/FEDIntegrity_EvF"

