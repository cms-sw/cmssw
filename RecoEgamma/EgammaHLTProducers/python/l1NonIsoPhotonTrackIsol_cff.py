import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaHLTProducers.hltPhotonTrackIsol_cfi import *
import copy
from RecoEgamma.EgammaHLTProducers.hltPhotonTrackIsol_cfi import *
l1NonIsoPhotonTrackIsol = copy.deepcopy(hltPhotonTrackIsol)
l1NonIsoPhotonTrackIsol.recoEcalCandidateProducer = 'l1NonIsoRecoEcalCandidate'
l1NonIsoPhotonTrackIsol.trackProducer = 'l1NonIsoEgammaRegionalCTFFinalFitWithMaterial'

