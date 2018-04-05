import FWCore.ParameterSet.Config as cms

dtpacker = cms.EDProducer("DTDigiToRawModule",
    useStandardFEDid = cms.untracked.bool(True),
    debugMode = cms.untracked.bool(False),
    digiColl = cms.InputTag("simMuonDTDigis"),
    minFEDid = cms.untracked.int32(770),
    maxFEDid = cms.untracked.int32(775)
)

import EventFilter.DTRawToDigi.dturospacker_cfi
_dturospacker = EventFilter.DTRawToDigi.dturospacker_cfi.dturospacker.clone()
from Configuration.Eras.Modifier_run2_DT_2018_cff import run2_DT_2018
run2_DT_2018.toReplaceWith(dtpacker, _dturospacker)
