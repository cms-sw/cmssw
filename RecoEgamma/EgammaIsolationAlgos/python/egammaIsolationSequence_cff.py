import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egammaIsoDeposits_cff import *
from RecoEgamma.EgammaIsolationAlgos.egammaIsoFromDeposits_cff import *
egammaIsolationSequence = cms.Sequence(egammaIsoDeposits*egammaIsoFromDeposits)

