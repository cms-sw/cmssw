import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.Castor.CastorCellReco_cfi import *
from RecoLocalCalo.Castor.CastorTowerReco_cfi import *
from RecoLocalCalo.Castor.CastorClusterReco_cfi import *
from RecoLocalCalo.Castor.CastorJetEgammaReco_cfi import *

CastorFullReco = cms.Sequence(CastorCellReco*CastorTowerReco*CastorClusterReco*CastorJetEgammaReco)

