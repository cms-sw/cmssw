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
#process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
#process.EcalTrivialConditionRetriever.producedEcalIntercalibConstants = True
# b) Getting conditions through frontier interface:
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = "CRAFT09_R_V4::All"
# c) Getting conditions through oracle interface:
#process.load("RecoLocalCalo.EcalRecProducers.getEcalConditions_orcoffint2r_cff")

#########################
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.dccWeightBuilder = cms.EDAnalyzer("EcalDccWeightBuilder",
  dcc1stSample = cms.int32(2),
  sampleToSkip = cms.int32(-1),
  mode = cms.string("computeWeights"),
  nDccWeights = cms.int32(6),
  inputWeights  = cms.vdouble(),
  dccWeightsWithIntercalib = cms.bool(False),
  writeToDB = cms.bool(False),
  writeToAsciiFile = cms.bool(True),
  writeToRootFile = cms.bool(True),
  dbSid = cms.string("cms_omds_lb"),
  dbUser = cms.string("cms_ecal_conf_test"),
  dbPassword = cms.untracked.string("file:conddb_passwd"),
  dbTag = cms.string("6-opt-weights"),
  dbVersion = cms.int32(0),
  sqlMode = cms.bool(True),
  asciiOutputFileName = cms.string("dccWeights.sql"),
  rootOutputFileName = cms.string("dccWeights.root"))

process.path = cms.Path(process.dccWeightBuilder)
