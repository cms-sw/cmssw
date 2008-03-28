import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaHLTProducers.hltHcalDoubleCone_cfi import *
import copy
from RecoEgamma.EgammaHLTProducers.hltHcalDoubleCone_cfi import *
l1NonIsoEMHcalDoubleCone = copy.deepcopy(hltHcalDoubleCone)
l1NonIsoEMHcalDoubleCone.recoEcalCandidateProducer = 'l1NonIsoRecoEcalCandidate'

