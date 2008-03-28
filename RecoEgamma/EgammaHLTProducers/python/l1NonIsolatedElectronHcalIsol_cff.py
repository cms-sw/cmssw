import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaHLTProducers.hltEgammaHcalIsol_cfi import *
import copy
from RecoEgamma.EgammaHLTProducers.hltEgammaHcalIsol_cfi import *
l1NonIsolatedElectronHcalIsol = copy.deepcopy(hltEgammaHcalIsol)
l1NonIsolatedElectronHcalIsol.recoEcalCandidateProducer = 'l1NonIsoRecoEcalCandidate'

