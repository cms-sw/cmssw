# ESProducer to provide and calculate and provide dataformats used by Hybrid emulator

import FWCore.ParameterSet.Config as cms

HybridDataFormats = cms.ESProducer("trklet::ProducerDataFormats")

fakeHybridDataFormatsSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('trklet::DataFormatsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)