## This configuration might NOT run directly using cmsRun ____________________||
## It is just to show example configuration of the MET filters for ICHEP 2012 ||
from PhysicsTools.PatAlgos.patTemplate_cfg import *

## The good primary vertex filter ____________________________________________||
process.primaryVertexFilter = cms.EDFilter(
    "VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
    filter = cms.bool(True)
    )

## The beam scraping filter __________________________________________________||
process.noscraping = cms.EDFilter(
    "FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(False),
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.25)
    )

## The iso-based HBHE noise filter ___________________________________________||
process.load('CommonTools.RecoAlgos.HBHENoiseFilter_cfi')

## The CSC beam halo tight filter ____________________________________________||
process.load('RecoMET.METAnalyzers.CSCHaloFilter_cfi')

## The HCAL laser filter _____________________________________________________||
process.load("RecoMET.METFilters.hcalLaserEventFilter_cfi")

## The ECAL dead cell trigger primitive filter _______________________________||
process.load('RecoMET.METFilters.EcalDeadCellTriggerPrimitiveFilter_cfi')

## The EE bad SuperCrystal filter ____________________________________________||
process.load('RecoMET.METFilters.eeBadScFilter_cfi')

## The ECAL laser correction filter
process.load('RecoMET.METFilters.ecalLaserCorrFilter_cfi')

## The Good vertices collection needed by the tracking failure filter ________||
process.goodVertices = cms.EDFilter(
  "VertexSelector",
  filter = cms.bool(False),
  src = cms.InputTag("offlinePrimaryVertices"),
  cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.rho < 2")
)

## The tracking failure filter _______________________________________________||
process.load('RecoMET.METFilters.trackingFailureFilter_cfi')

process.filtersSeq = cms.Sequence(
   process.primaryVertexFilter *
   process.noscraping *
   process.HBHENoiseFilter *
   process.CSCTightHaloFilter *
   process.hcalLaserEventFilter *
   process.EcalDeadCellTriggerPrimitiveFilter *
   process.goodVertices * process.trackingFailureFilter *
   process.eeBadScFilter *
   process.ecalLaserCorrFilter
)
