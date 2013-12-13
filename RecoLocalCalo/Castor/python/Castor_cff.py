import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.Castor.CastorTowerReco_cfi import *
from RecoJets.JetProducers.ak8CastorJets_cfi import *
from RecoJets.JetProducers.ak8CastorJetID_cfi import *

CastorFullReco = cms.Sequence(CastorTowerReco*ak8BasicJets*ak8CastorJetID)

