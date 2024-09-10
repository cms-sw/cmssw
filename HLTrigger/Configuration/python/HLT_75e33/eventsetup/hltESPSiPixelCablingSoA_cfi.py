import FWCore.ParameterSet.Config as cms

def _addProcessSiPixelCablingAlpaka(process):
    process.hltESPSiPixelCablingSoA = cms.ESProducer('SiPixelCablingSoAESProducer@alpaka', 
        CablingMapLabel = cms.string(''),
        UseQualityInfo = cms.bool(False),
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
    )

from Configuration.ProcessModifiers.alpaka_cff import alpaka
modifyConfigurationForAlpakaSiPixelCabling_ = alpaka.makeProcessModifier(_addProcessSiPixelCablingAlpaka)
