import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.Castor.CastorCellReco_cfi import *
from RecoLocalCalo.Castor.CastorTowerReco_cfi import *
from RecoLocalCalo.Castor.CastorClusterReco_cfi import *
from RecoLocalCalo.Castor.CastorJetEgammaReco_cfi import *
from RecoLocalCalo.Castor.CastorFastjetReco_cfi import * 

CastorFastjetRecoAntiKt07.src	   = cms.InputTag("CastorTowerReco")
CastorFastjetRecoAntiKt07.jetType      = cms.string("BasicJet")
# minimum jet pt
CastorFastjetRecoAntiKt07.jetPtMin       = cms.double(0.0)
# minimum calo tower input et
CastorFastjetRecoAntiKt07.inputEtMin     = cms.double(0.0)
# minimum calo tower input energy
CastorFastjetRecoAntiKt07.inputEMin      = cms.double(0.0)
# primary vertex correction
CastorFastjetRecoAntiKt07.doPVCorrection = cms.bool(False)

#CastorFastjetReco = cms.Sequence(CastorFastjetRecoKt+CastorFastjetRecoSISCone+CastorFastjetRecoKtGen+CastorFastjetRecoSISConeGen)
#CastorFastjetReco = cms.Sequence(CastorFastjetRecoKt+CastorFastjetRecoSISCone)
#CastorClusterReco = cms.Sequence(CastorClusterRecoCustomKt+CastorClusterRecoKt+CastorClusterRecoSISCone)
#CastorJetEgammaReco = cms.Sequence(CastorJetEgammaRecoCustomKt+CastorJetEgammaRecoKt+CastorJetEgammaRecoSISCone)

CastorFullReco = cms.Sequence(CastorCellReco*CastorTowerReco*CastorFastjetRecoAntiKt07*CastorClusterRecoAntiKt07*CastorJetEgammaRecoAntiKt07)

