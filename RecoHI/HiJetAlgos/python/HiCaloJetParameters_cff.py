import FWCore.ParameterSet.Config as cms

## import and modify standard CaloJetParameters
import RecoJets.JetProducers.CaloJetParameters_cfi
HiCaloJetDefaults = RecoJets.JetProducers.CaloJetParameters_cfi.CaloJetParameters.clone(
    doPUOffsetCorr = True,
    doAreaFastjet = True,
    doRhoFastjet = False,
    doPVCorrection = False,
    jetPtMin = 10,
    Ghost_EtaMax = 6.5
   
)


## add non-uniform fastjet settings
HiCaloJetParameters = cms.PSet(
    HiCaloJetDefaults,
    doFastJetNonUniform = cms.bool(True),
    puCenters = cms.vdouble(-5,-4,-3,-2,-1,0,1,2,3,4,5),
    puWidth = cms.double(0.8),
    nExclude = cms.uint32(2),
    dropZeros = cms.bool(True),
    addNegative = cms.bool(True),
    addNegativesFromCone = cms.bool(False),
    infinitesimalPt = cms.double(0.005)
)

## default settings for various pileup subtractors
MultipleAlgoIteratorBlock = cms.PSet(
    subtractorName = cms.string("MultipleAlgoIterator"),
    sumRecHits = cms.bool(False)
)

ParametrizedSubtractorBlock = cms.PSet(
    subtractorName = cms.string("ParametrizedSubtractorBlock"),
    sumRecHits = cms.bool(False),
    interpolate = cms.bool(False)
)

