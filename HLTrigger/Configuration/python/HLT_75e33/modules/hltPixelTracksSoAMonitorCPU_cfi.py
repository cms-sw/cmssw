import FWCore.ParameterSet.Config as cms

hltPixelTracksSoAMonitorCPU =  cms.EDProducer('SiPixelMonitorTrackSoA',
                                              pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoASerialSync'),
                                              topFolderName = cms.string('HLT/HeterogeneousMonitoring/PixelTracksCPU'),
                                              qualityDefinitions = cms.vstring(
                                                  'loose',
                                                  'highPurity'
                                              ))
