import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egammaTrackExtractorBlocks_cff import *
from RecoEgamma.EgammaIsolationAlgos.egammaCalExtractorBlocks_cff import *
from RecoEgamma.EgammaIsolationAlgos.eleIsoModules_cff import *

eleIsoDeposits = cms.Sequence(*eleIsoDepositTk*eleIsoDepositEcalFromHits*eleIsoDepositHcalFromHits)

