import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.MCJetCorrections152_cff import *
from JetMETCorrections.Configuration.L5FlavorCorrections_cff import *
from JetMETCorrections.Type1MET.MetType1Corrections_cff import *
patJetMETCorrections = cms.Sequence(corMetType1Icone5)

