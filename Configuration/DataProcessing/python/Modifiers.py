import FWCore.ParameterSet.Config as cms

modifyPPData = cms.ModifierChain()

modifyCosmicData = cms.ModifierChain()

import RecoVertex.BeamSpotProducer.Modifiers as _modsBS
modifyExpress = cms.ModifierChain(modifyPPData, _modsBS.offlineToOnlineBeamSpotSwap)

modifyCommonHI = cms.ModifierChain()

modifyExpressHI = cms.ModifierChain(modifyCommonHI, _modsBS.offlineToOnlineBeamSpotSwap)

