import FWCore.ParameterSet.Config as cms

from RecoBTag.TensorFlow.pfDeepFlavourTagInfos_cfi import pfDeepFlavourTagInfos
from RecoBTag.TensorFlow.pfDeepFlavourJetTags_cfi import pfDeepFlavourJetTags
from RecoBTag.TensorFlow.pfNegativeDeepFlavourTagInfos_cfi import pfNegativeDeepFlavourTagInfos
from RecoBTag.TensorFlow.pfNegativeDeepFlavourJetTags_cfi import pfNegativeDeepFlavourJetTags
from RecoBTag.TensorFlow.pfDeepDoubleBTagInfos_cfi import pfDeepDoubleBTagInfos
from RecoBTag.TensorFlow.pfDeepDoubleBJetTags_cfi import pfDeepDoubleBJetTags
from CommonTools.PileupAlgos.Puppi_cff import puppi
from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import primaryVertexAssociation

# This task is not used, useful only if we run DeepFlavour from RECO
# jets (RECO/AOD)
pfDeepFlavourTask = cms.Task(puppi, primaryVertexAssociation,
                             pfDeepFlavourTagInfos, pfDeepFlavourJetTags)
