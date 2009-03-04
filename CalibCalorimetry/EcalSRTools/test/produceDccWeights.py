import FWCore.ParameterSet.Config as cms

process = cms.Process("ProdDCCWeights")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

# ecal mapping
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.eegeom = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalMappingRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

# Conditions:
#
# a) Getting hardcoded conditions the same used for standard digitization:
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
# b) Getting conditions through frontier interface:
# process.load("RecoLocalCalo.EcalRecProducers.getEcalConditions_frontier_cff")
# c) Getiing conditions through oracle interface:
#process.load("RecoLocalCalo.EcalRecProducers.getEcalConditions_orcoffint2r_cff.py")

#########################
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.dccWeightBuilder = cms.EDAnalyzer("EcalDccWeightBuilder",
  dcc1stSample = cms.int32(2),
  sampleToSkip = cms.int32(-1),
  nDccWeights = cms.int32(6),
  dccWeightsWithIntercalib = cms.bool(True),
  writeToDB = cms.bool(False),
  writeToAsciiFile = cms.bool(True),
  writeToRootFile = cms.bool(True),
  asciiOutputFileName = cms.string("dccWeights.txt"),
  rootOutputFileName = cms.string("dccWeights.root"))

process.path = cms.Path(process.dccWeightBuilder)
