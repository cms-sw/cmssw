import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaHLTProducers.hltEgammaHcalIsol_cfi import *
import copy
from RecoEgamma.EgammaHLTProducers.hltEgammaHcalIsol_cfi import *
l1IsolatedPhotonHcalIsol = copy.deepcopy(hltEgammaHcalIsol)
l1IsolatedPhotonHcalIsol.recoEcalCandidateProducer = 'l1IsoRecoEcalCandidate'
#  InputTag hbRecHitProducer      = hbhereco
#  InputTag hfRecHitProducer      = hfreco
l1IsolatedPhotonHcalIsol.egHcalIsoPtMin = 0.
l1IsolatedPhotonHcalIsol.egHcalIsoConeSize = 0.3

