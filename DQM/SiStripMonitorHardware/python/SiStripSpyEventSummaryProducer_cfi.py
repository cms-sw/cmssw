import FWCore.ParameterSet.Config as cms

SiStripSpyEventSummary = cms.EDProducer('SiStripSpyEventSummaryProducer',
  RawDataTag = cms.InputTag('source'),
  RunType = cms.uint32(2) #Pedestals, see DataFormats/SiStripCommon/interface/ConstantsForRunType.h
)
