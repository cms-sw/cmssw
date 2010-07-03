import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.Castor.CastorTowerReco_cfi import *
from RecoJets.JetProducers.ak7BasicJets_cfi import *
from RecoJets.JetProducers.ak7CastorJetID_cfi import *

CastorFullReco = cms.Sequence(CastorTowerReco*ak7BasicJets*ak7CastorJetID)

