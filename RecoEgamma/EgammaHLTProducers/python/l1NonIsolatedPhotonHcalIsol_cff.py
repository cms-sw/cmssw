import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaHLTProducers.hltEgammaHcalIsol_cfi import *
import copy
from RecoEgamma.EgammaHLTProducers.hltEgammaHcalIsol_cfi import *
l1NonIsolatedPhotonHcalIsol = copy.deepcopy(hltEgammaHcalIsol)
l1NonIsolatedPhotonHcalIsol.recoEcalCandidateProducer = 'l1NonIsoRecoEcalCandidate'
#  InputTag hbRecHitProducer      = hbhereco
#  InputTag hfRecHitProducer      = hfreco
l1NonIsolatedPhotonHcalIsol.egHcalIsoPtMin = 0.
l1NonIsolatedPhotonHcalIsol.egHcalIsoConeSize = 0.3

