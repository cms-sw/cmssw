import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PileupJetIDParams_cfi import * 

#_stdalgos_4x = cms.VPSet(full,   cutbased,PhilV1)
_stdalgos_5x = cms.VPSet(full_5x,cutbased,PhilV1)

#_chsalgos_4x = cms.VPSet(full,   cutbased) 
_chsalgos_5x = cms.VPSet(full_5x_chs,cutbased)
_chsalgos_74x = cms.VPSet(full_74x_chs,cutbased)
_chsalgos_76x = cms.VPSet(full_76x_chs,cutbased)
_chsalgos_80x = cms.VPSet(full_80x_chs,cutbased)
_chsalgos_81x = cms.VPSet(full_81x_chs,cutbased)

_stdalgos    = _chsalgos_81x

# Calculate+store variables and run MVAs
pileupJetId = cms.EDProducer('PileupJetIdProducer',
     produceJetIds = cms.bool(True),
     jetids = cms.InputTag(""),
     runMvas = cms.bool(True),
     jets = cms.InputTag("ak4PFJetsCHS"),
     vertexes = cms.InputTag("offlinePrimaryVertices"),
     algos = cms.VPSet(_stdalgos),
     rho     = cms.InputTag("fixedGridRhoFastjetAll"),
     jec     = cms.string("AK4PFchs"),
     applyJec = cms.bool(True),
     inputIsCorrected = cms.bool(False),
     residualsFromTxt = cms.bool(False),
     usePuppi = cms.bool(False),
#     residualsTxt     = cms.FileInPath("RecoJets/JetProducers/data/download.url") # must be an existing file
)

# Calculate variables, but don't run MVAs
pileupJetIdCalculator = pileupJetId.clone(
    runMvas = cms.bool(False),
    algos = cms.VPSet(cutbased),
)

# Run MVAs on precalculated variables
pileupJetIdEvaluator = pileupJetId.clone(
    produceJetIds = cms.bool(False),
    jetids = cms.InputTag("pileupJetIdCalculator"),
)

pileUpJetIDTask = cms.Task(pileupJetId,
                           pileupJetIdCalculator,
                           pileupJetIdEvaluator
)
