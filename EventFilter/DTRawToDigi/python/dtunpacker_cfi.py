import FWCore.ParameterSet.Config as cms
import EventFilter.DTRawToDigi.dtUnpackingModule_cfi

# Module for DT data unpacking: produces a DTDigiCollection and - on demand - 
# a DTLocalTriggerCollection
muonDTDigis = EventFilter.DTRawToDigi.dtUnpackingModule_cfi.dtUnpackingModule.clone()
muonDTDigis.dataType = cms.string('DDU')
muonDTDigis.inputLabel = cms.InputTag('rawDataCollector')
muonDTDigis.useStandardFEDid = cms.bool(True)
muonDTDigis.dqmOnly = cms.bool(False)
muonDTDigis.readOutParameters = cms.PSet(
    debug = cms.untracked.bool(False),
    rosParameters = cms.PSet(
        writeSC = cms.untracked.bool(True),
        readingDDU = cms.untracked.bool(True),
        performDataIntegrityMonitor = cms.untracked.bool(False),
        readDDUIDfromDDU = cms.untracked.bool(True),
        debug = cms.untracked.bool(False),
        localDAQ = cms.untracked.bool(False)
    ),
    localDAQ = cms.untracked.bool(False),
    performDataIntegrityMonitor = cms.untracked.bool(False)
)

import EventFilter.DTRawToDigi.dturosunpacker_cfi
_dturosunpacker = EventFilter.DTRawToDigi.dturosunpacker_cfi.dturosunpacker.clone()
from Configuration.Eras.Modifier_run2_DT_2018_cff import run2_DT_2018
run2_DT_2018.toReplaceWith(muonDTDigis, _dturosunpacker)
