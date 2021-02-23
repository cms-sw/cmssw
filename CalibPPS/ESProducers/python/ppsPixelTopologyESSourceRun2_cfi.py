import FWCore.ParameterSet.Config as cms

ppsPixelTopologyESSource = cms.ESSource('PPSPixelTopologyESSource',
  RunType = cms.string('Run2'),
  PitchSimY = cms.double(0.15),
  PitchSimX = cms.double(0.1),
  thickness = cms.double(0.23),
  noOfPixelSimX = cms.int32(160),
  noOfPixelSimY = cms.int32(156),
  noOfPixels = cms.int32(24960),
  simXWidth = cms.double(16.6),
  simYWidth = cms.double(24.4),
  deadEdgeWidth = cms.double(0.2),
  activeEdgeSigma = cms.double(0.02),
  physActiveEdgeDist = cms.double(0.15),
  appendToDataLabel = cms.string('')
)
