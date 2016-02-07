import FWCore.ParameterSet.Config as cms

l1tCaloLayer1Digis = cms.EDProducer(
    'L1TCaloLayer1RawToDigi',
    fedRawDataLabel = cms.InputTag("rawDataCollector"),
    FEDIDs = cms.vint32(1354, 1356, 1358),
    verbose = cms.bool(False)
    )
