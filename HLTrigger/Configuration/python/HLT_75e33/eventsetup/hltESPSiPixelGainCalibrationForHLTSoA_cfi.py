import FWCore.ParameterSet.Config as cms

def _addProcessSiPixelGainCalibrationAlpaka(process):
    process.hltESPSiPixelGainCalibrationForHLTSoA = cms.ESProducer('SiPixelGainCalibrationForHLTSoAESProducer@alpaka',
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
    )

from Configuration.ProcessModifiers.alpaka_cff import alpaka
modifyConfigurationForAlpakaSiPixelGainCalibration_ = alpaka.makeProcessModifier(_addProcessSiPixelGainCalibrationAlpaka)
