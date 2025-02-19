import FWCore.ParameterSet.Config as cms

from FastSimulation.ForwardDetectors.CastorFastTowerProducer_cfi import *
from FastSimulation.ForwardDetectors.CastorFastClusterProducer_cfi import *

#from RecoLocalCalo.Castor.CastorClusterReco_cfi import *
#CastorClusterReco.inputprocess = "CastorFastTowerReco"
#from RecoLocalCalo.Castor.CastorJetEgammaReco_cfi import *
#CastorJetEgammaReco.fastsim = True

CastorFastReco = cms.Sequence(CastorFastTowerReco*CastorFastClusterReco)
