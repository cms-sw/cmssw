import FWCore.ParameterSet.Config as cms

import copy
from EventFilter.CSCRawToDigi.cscUnpacker_cfi import *
muonCSCDigis = copy.deepcopy(muonCSCDigis)
CSCRawToDigi = cms.Sequence(muonCSCDigis)
muonCSCDigis.InputObjects = 'rawDataCollector'
muonCSCDigis.UseExaminer = False

