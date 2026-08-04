import FWCore.ParameterSet.Config as cms

hltPixelTracksSoAMonitorGPU =  cms.EDProducer('SiPixelMonitorTrackSoA',
                                              pixelTrackSrc = cms.InputTag('hltPhase2PixelTrackTorchHighPuritySelector'),
                                              topFolderName = cms.string('HLT/HeterogeneousMonitoring/PixelTracksGPU'),
                                              qualityDefinitions = cms.vstring(
                                                  'highPurity'
                                              ))
