import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egammaTrackExtractorBlocks_cff import *
from RecoEgamma.EgammaIsolationAlgos.egammaCalExtractorBlocks_cff import *
from RecoEgamma.EgammaIsolationAlgos.gamIsoModules_cff import *

gamIsoDeposits = cms.Sequence(gamIsoDepositTk*gamIsoDepositEcalFromHits*gamIsoDepositHcalFromHits)

