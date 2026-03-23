import FWCore.ParameterSet.Config as cms

hltPixelTracksSoAMonitorGPU =  cms.EDProducer('SiPixelMonitorTrackSoA',
                                              pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoA'),
                                              topFolderName = cms.string('HLT/HeterogeneousMonitoring/PixelTracksGPU'),
                                              qualityDefinitions = cms.vstring(
                                                  'loose',
                                                  'highPurity'
                                              ))
