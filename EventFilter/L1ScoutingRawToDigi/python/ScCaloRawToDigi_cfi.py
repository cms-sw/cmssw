import FWCore.ParameterSet.Config as cms

# default parameters for the DMA input source
dmaDataModeParameters = cms.PSet(
  dataSourceMode=cms.string("DMA"),
  dmaSourceId=cms.int32(2)
)

# default parameters for the TCP source
tcpDataModeParameters = cms.PSet(
  dataSourceMode=cms.string("TCP"),
  jetSourceIdList = cms.vint32(22),
  eGammaSourceIdList = cms.vint32(23),
  tauSourceIdList = cms.vint32(25),
  etSumSourceIdList = cms.vint32(24),
)

ScCaloUnpacker = cms.EDProducer("ScCaloRawToDigi",
  srcInputTag = cms.InputTag("rawDataCollector"),
  # DMA / TCP mode configs
  dataSource = tcpDataModeParameters,
  # unpack the full set of energy sums
  enableAllSums = cms.bool(True)
)

