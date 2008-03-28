import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaHLTProducers.hltPhotonTrackIsol_cfi import *
import copy
from RecoEgamma.EgammaHLTProducers.hltPhotonTrackIsol_cfi import *
l1IsoPhotonTrackIsol = copy.deepcopy(hltPhotonTrackIsol)
l1IsoPhotonTrackIsol.recoEcalCandidateProducer = 'l1IsoRecoEcalCandidate'
l1IsoPhotonTrackIsol.trackProducer = 'l1IsoEgammaRegionalCTFFinalFitWithMaterial'

