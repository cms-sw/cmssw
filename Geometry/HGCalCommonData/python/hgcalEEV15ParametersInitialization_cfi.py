import FWCore.ParameterSet.Config as cms

hgcalEEParametersInitialize = cms.ESProducer('HGCalParametersESModule',
  name = cms.string('HGCalEELayer'),
  name2 = cms.string('HGCalEESensitive'),
  nameW = cms.string('HGCalEEWafer'),
  nameC = cms.string('HGCalEESensitive'),
  nameT = cms.string('HGCal'),
  nameX = cms.string('HGCalEESensitive'),
  fromDD4hep = cms.bool(False),
  appendToDataLabel = cms.string('')
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hgcalEEParametersInitialize,
                fromDD4hep = True
)
