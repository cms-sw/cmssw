import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorModule.dtDataIntegrityTask_cfi import *
dtDataIntegrityTask.processingMode = 'HLT'
dtDataIntegrityTask.fedIntegrityFolder = 'DT/FEDIntegrity_EvF'

