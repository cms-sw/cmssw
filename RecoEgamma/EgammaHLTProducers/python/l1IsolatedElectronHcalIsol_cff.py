import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaHLTProducers.hltEgammaHcalIsol_cfi import *
import copy
from RecoEgamma.EgammaHLTProducers.hltEgammaHcalIsol_cfi import *
l1IsolatedElectronHcalIsol = copy.deepcopy(hltEgammaHcalIsol)
l1IsolatedElectronHcalIsol.recoEcalCandidateProducer = 'l1IsoRecoEcalCandidate'

