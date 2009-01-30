import FWCore.ParameterSet.Config as cms
from L1TriggerOffline.L1Analyzer.BSCTrigger_cfi import *
from Configuration.StandardSequences.SimL1Emulator_cff import *
simGtDigis.TechnicalTriggersInputTag = 'bscTrigger'
simGtDigis.ReadTechnicalTriggerRecords = cms.bool ( True )
