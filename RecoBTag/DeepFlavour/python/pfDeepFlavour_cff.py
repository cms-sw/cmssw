import FWCore.ParameterSet.Config as cms

from RecoBTag.DeepFlavour.pfDeepFlavourTagInfos_cfi import pfDeepFlavourTagInfos
from RecoBTag.DeepFlavour.pfDeepFlavourJetTags_cfi import pfDeepFlavourJetTags
from CommonTools.PileupAlgos.Puppi_cff import puppi
from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import primaryVertexAssociation


pfDeepFlavourTask = cms.Task(puppi, primaryVertexAssociation,
                             pfDeepFlavourTagInfos, pfDeepFlavourJetTags)
