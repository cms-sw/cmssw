import FWCore.ParameterSet.Config as cms

from RecoBTag.DeepFlavour.DeepFlavourTagInfos_cfi import pfDeepFlavourTagInfos
from RecoBTag.DeepFlavour.DeepFlavourJetTags_cfi import pfDeepFlavourJetTags
from CommonTools.PileupAlgos.Puppi_cff import puppi
from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import primaryVertexAssociation


pfDeepFlavourTaskNew = cms.Task(puppi, primaryVertexAssociation,
                                pfDeepFlavourTagInfos, pfDeepFlavourJetTags)
