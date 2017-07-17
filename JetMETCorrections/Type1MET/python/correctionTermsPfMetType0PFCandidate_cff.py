import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Type1MET.pfMETCorrectionType0_cfi import *

##____________________________________________________________________________||
corrPfMetType0PfCand = pfMETcorrType0.clone()

##____________________________________________________________________________||
correctionTermsPfMetType0PFCandidate = cms.Sequence(
    type0PFMEtCorrectionPFCandToVertexAssociation +
    corrPfMetType0PfCand
    )

correctionTermsPfMetType0PFCandidateForValidation = cms.Sequence(
    type0PFMEtCorrectionPFCandToVertexAssociationForValidation +
    corrPfMetType0PfCand
    )

##____________________________________________________________________________||
