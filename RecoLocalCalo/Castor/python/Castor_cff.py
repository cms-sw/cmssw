import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.Castor.CastorCellReco_cfi import *
from RecoLocalCalo.Castor.CastorTowerReco_cfi import *
from RecoLocalCalo.Castor.CastorTowerCandidateReco_cfi import *
from RecoLocalCalo.Castor.CastorClusterReco_cfi import *
from RecoLocalCalo.Castor.CastorJetEgammaReco_cfi import *
from RecoLocalCalo.Castor.CastorFastjetReco_cfi import * 

#CastorFastjetReco = cms.Sequence(CastorFastjetRecoKt+CastorFastjetRecoSISCone+CastorFastjetRecoKtGen+CastorFastjetRecoSISConeGen)
CastorFastjetReco = cms.Sequence(CastorFastjetRecoKt+CastorFastjetRecoSISCone)
CastorClusterReco = cms.Sequence(CastorClusterRecoCustomKt+CastorClusterRecoKt+CastorClusterRecoSISCone)
CastorJetEgammaReco = cms.Sequence(CastorJetEgammaRecoCustomKt+CastorJetEgammaRecoKt+CastorJetEgammaRecoSISCone)

CastorFullReco = cms.Sequence(CastorCellReco*CastorTowerReco*CastorTowerCandidateReco*CastorFastjetReco*CastorClusterReco*CastorJetEgammaReco)

