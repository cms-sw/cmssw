import FWCore.ParameterSet.Config as cms

hltPixelTracksSoACompareGPUvsCPU = cms.EDProducer('SiPixelCompareTracksSoA',
                                                  pixelTrackReferenceSoA = cms.InputTag('hltPhase2PixelTracksSoASerialSync'),
                                                  pixelTrackTargetSoA = cms.InputTag('hltPhase2PixelTracksSoA'),
                                                  topFolderName = cms.string('HLT/HeterogeneousComparisons/pixelTracksSoA'),
                                                  useQualityCut = cms.bool(True),
                                                  minQuality = cms.string('loose'),
                                                  deltaR2cut = cms.double(0.0004),
                                                  )
