import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.Castor.CastorTowerReco_cfi import *
from RecoJets.JetProducers.ak5CastorJets_cfi import *
from RecoJets.JetProducers.ak5CastorJetID_cfi import *
from RecoJets.JetProducers.ak7CastorJets_cfi import *
from RecoJets.JetProducers.ak7CastorJetID_cfi import *

CastorFullReco = cms.Sequence(CastorTowerReco*
                              ak5CastorJets*ak5CastorJetID*
			      ak7CastorJets*ak7CastorJetID)
