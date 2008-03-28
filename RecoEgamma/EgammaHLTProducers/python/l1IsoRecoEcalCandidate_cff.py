import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaHLTProducers.hltRecoEcalCandidate_cfi import *
import copy
from RecoEgamma.EgammaHLTProducers.hltRecoEcalCandidate_cfi import *
l1IsoRecoEcalCandidate = copy.deepcopy(hltRecoEcalCandidate)
l1IsoRecoEcalCandidate.scHybridBarrelProducer = 'correctedHybridSuperClustersL1Isolated'
l1IsoRecoEcalCandidate.scIslandEndcapProducer = 'correctedEndcapSuperClustersWithPreshowerL1Isolated'

