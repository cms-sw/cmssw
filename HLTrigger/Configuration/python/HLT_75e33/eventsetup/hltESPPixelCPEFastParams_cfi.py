import FWCore.ParameterSet.Config as cms

def _addProcessPixelCPEFastParamsPhase2(process):
    process.hltESPPixelCPEFastParamsPhase2 = cms.ESProducer('PixelCPEFastParamsESProducerAlpakaPhase2@alpaka', 
        ComponentName = cms.string("PixelCPEFastParamsPhase2"),
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(backend = cms.untracked.string('')
    )
)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
modifyConfigurationForAlpakaPixelCPEFastParamsPhase2_ = alpaka.makeProcessModifier(_addProcessPixelCPEFastParamsPhase2)
