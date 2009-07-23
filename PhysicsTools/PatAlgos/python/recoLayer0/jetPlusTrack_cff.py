import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.ZSPJetCorrections219_cff import *
from JetMETCorrections.Configuration.JetPlusTrackCorrections_cff import *

jptCaloJets = cms.Sequence(
    ZSPJetCorrections *
    JetPlusTrackCorrections
    )
