import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run2_jme_2016_cff import run2_jme_2016
from Configuration.Eras.Modifier_run2_jme_2017_cff import run2_jme_2017
from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016

from RecoJets.JetProducers.PileupJetIDParams_cfi import *

_chsalgos_106X_UL17 = cms.VPSet(full_106x_UL17_chs,cutbased)
_chsalgos_106X_UL18 = cms.VPSet(full_106x_UL18_chs,cutbased)
_chsalgos_106X_UL16 = cms.VPSet(full_106x_UL16_chs,cutbased)
_chsalgos_106X_UL16APV = cms.VPSet(full_106x_UL16APV_chs,cutbased)

_stdalgos    = _chsalgos_106X_UL18

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
     srcConstituentWeights = cms.InputTag(""),
#     residualsTxt     = cms.FileInPath("RecoJets/JetProducers/data/download.url") # must be an existing file
)

# set algos for re-miniAOD campaign
(run2_miniAOD_UL & run2_jme_2017).toModify(pileupJetId, algos = _chsalgos_106X_UL17)
(run2_miniAOD_UL & run2_jme_2016 & ~tracker_apv_vfp30_2016).toModify(pileupJetId, algos = _chsalgos_106X_UL16)
(run2_miniAOD_UL & run2_jme_2016 & tracker_apv_vfp30_2016).toModify(pileupJetId, algos = _chsalgos_106X_UL16APV)

# Calculate variables, but don't run MVAs
pileupJetIdCalculator = pileupJetId.clone(
    runMvas = False,
    algos   = cms.VPSet(cutbased),
)

# Run MVAs on precalculated variables
pileupJetIdEvaluator = pileupJetId.clone(
    produceJetIds = False,
    jetids        = "pileupJetIdCalculator",
)

pileUpJetIDTask = cms.Task(pileupJetId,
                           pileupJetIdCalculator,
                           pileupJetIdEvaluator
)

#
#
#
_puppiV18algos_133X_Winter24 = cms.VPSet(full_133x_Winter24_puppi_v18_wp,cutbased)

pileupJetIdPuppi = pileupJetId.clone(
    jets = "ak4PFJetsPuppi",
    jec  = "AK4PFPuppi",
    srcConstituentWeights = "puppi",
    algos = cms.VPSet(_puppiV18algos_133X_Winter24),
)

pileUpJetIDPuppiTask = cms.Task(pileupJetIdPuppi)

