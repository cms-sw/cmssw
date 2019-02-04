import FWCore.ParameterSet.Config as cms

from RecoBTag.FeatureTools.pfDeepFlavourTagInfos_cfi import pfDeepFlavourTagInfos
from RecoBTag.FeatureTools.pfNegativeDeepFlavourTagInfos_cfi import pfNegativeDeepFlavourTagInfos
from RecoBTag.FeatureTools.pfDeepDoubleXTagInfos_cfi import pfDeepDoubleXTagInfos

from RecoBTag.TensorFlow.pfDeepFlavourJetTags_cfi import pfDeepFlavourJetTags
from RecoBTag.TensorFlow.pfNegativeDeepFlavourJetTags_cfi import pfNegativeDeepFlavourJetTags
from CommonTools.PileupAlgos.Puppi_cff import puppi
from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import primaryVertexAssociation

# This task is not used, useful only if we run DeepFlavour from RECO
# jets (RECO/AOD)
pfDeepFlavourTask = cms.Task(puppi, primaryVertexAssociation,
                             pfDeepFlavourTagInfos, pfDeepFlavourJetTags)
