import FWCore.ParameterSet.Config as cms

hcalhistos = cms.EDProducer("HcalHistogramRawToDigi",
    InputLabel = cms.InputTag("rawDataCollector"),
    # Number of the first HCAL FED.  If this is not specified, the
    # default from FEDNumbering is used.
    HcalFirstFED = cms.untracked.int32(20),
    # FED numbers to unpack.  If this is not specified, all FEDs from
    # FEDNumbering will be unpacked.
    FEDs = cms.untracked.vint32(20)
)


