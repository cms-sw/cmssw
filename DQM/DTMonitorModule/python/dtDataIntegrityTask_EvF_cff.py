import FWCore.ParameterSet.Config as cms

dtunpacker = cms.EDProducer("DTUnpackingModule",
    dataType = cms.string('DDU'),
    inputLabel = cms.InputTag('rawDataCollector'),
    fedbyType = cms.bool(False),
    useStandardFEDid = cms.bool(False),
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

import EventFilter.DTRawToDigi.dturosunpacker_cfi
_dturosunpacker = EventFilter.DTRawToDigi.dturosunpacker_cfi.dturosunpacker.clone()
from Configuration.Eras.Modifier_run2_DT_2018_cff import run2_DT_2018
run2_DT_2018.toReplaceWith(dtunpacker, _dturosunpacker)


