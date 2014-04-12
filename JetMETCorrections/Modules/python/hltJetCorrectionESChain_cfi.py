# HLT template for JetCorrectionESChain

import FWCore.ParameterSet.Config as cms

# L2L3 correction
hltJetCorrectionESChain = cms.ESProducer( 'JetCorrectionESChain',
    appendToDataLabel = cms.string( '' ),
    correctors = cms.vstring(
      'hltESSL2RelativeCorrectionService',
      'hltESSL3AbsoluteCorrectionService'
    )
)
