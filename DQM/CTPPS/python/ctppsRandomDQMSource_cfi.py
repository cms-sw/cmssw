import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ctppsRandomDQMSource = DQMEDAnalyzer('CTPPSRandomDQMSource',
    tagRPixDigi = cms.untracked.InputTag("ctppsPixelDigisAlCaRecoProducer", ""),
    RPStatusWord = cms.untracked.uint32(0x8008), # rpots in readout:220_fr_hr; 210_fr_hr
)