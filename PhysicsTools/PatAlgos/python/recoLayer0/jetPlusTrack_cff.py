import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.ZSPJetCorrections219_cff import *
from JetMETCorrections.Configuration.JetPlusTrackCorrections_cff import *

jptCaloJets = cms.Sequence(
    ZSPJetCorrectionsIcone5 *
    ZSPJetCorrectionsSisCone5 *
    ZSPJetCorrectionsAntiKt5 *
    JetPlusTrackCorrectionsIcone5 *
    JetPlusTrackCorrectionsSisCone5 *
    JetPlusTrackCorrectionsAntiKt5
    )
