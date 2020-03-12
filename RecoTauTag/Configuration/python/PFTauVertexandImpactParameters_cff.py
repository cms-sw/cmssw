import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFTauPrimaryVertexProducer_cfi      import *
from RecoTauTag.RecoTau.PFTauSecondaryVertexProducer_cfi    import *
from RecoTauTag.RecoTau.PFTauTransverseImpactParameters_cfi import *

PFTauVertexandImpactParametersTask = cms.Task(PFTauPrimaryVertexProducer,PFTauSecondaryVertexProducer,PFTauTransverseImpactParameters)
PFTauVertexandImpactParametersSeq = cms.Sequence(PFTauVertexandImpactParametersTask)
