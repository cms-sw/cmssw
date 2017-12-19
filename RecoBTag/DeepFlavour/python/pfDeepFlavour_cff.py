import FWCore.ParameterSet.Config as cms

from RecoBTag.DeepFlavour.pfDeepFlavourTagInfos_cfi import pfDeepFlavourTagInfos
from RecoBTag.DeepFlavour.pfDeepFlavourJetTags_cfi import pfDeepFlavourJetTags
from RecoBTag.DeepFlavour.pfNegativeDeepFlavourTagInfos_cfi import pfNegativeDeepFlavourTagInfos
from RecoBTag.DeepFlavour.pfNegativeDeepFlavourJetTags_cfi import pfNegativeDeepFlavourJetTags
from CommonTools.PileupAlgos.Puppi_cff import puppi
from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import primaryVertexAssociation

# This task is not used, useful only if we run DeepFlavour from RECO
# jets (RECO/AOD)
pfDeepFlavourTask = cms.Task(puppi, primaryVertexAssociation,
                             pfDeepFlavourTagInfos, pfDeepFlavourJetTags)
