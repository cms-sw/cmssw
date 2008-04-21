import FWCore.ParameterSet.Config as cms

from FastSimulation.HighLevelTrigger.common.RecoLocalTracker_cff import *
from FastSimulation.HighLevelTrigger.common.RecoLocalCalo_cff import *
from FastSimulation.HighLevelTrigger.common.EcalRegionalReco_cff import *
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
from FastSimulation.HighLevelTrigger.common.HLTFastReco_cff import *
from FastSimulation.HighLevelTrigger.common.HLTEndpath_cff import *
doLocalPixel = cms.Sequence(pixeltrackerlocalreco)
doLocalStrip = cms.Sequence(striptrackerlocalreco)
doLocalTracker = cms.Sequence(doLocalPixel+doLocalStrip)
hcalLocalRecoWithoutHO = cms.Sequence(hbhereco+hfreco)
doLocalEcal_nopreshower = cms.Sequence(ecalLocalRecoSequence_nopreshower)
doLocalEcal = cms.Sequence(ecalLocalRecoSequence)
doLocalHcal = cms.Sequence(hcalLocalRecoSequence)
doLocalHcalWithoutHO = cms.Sequence(hcalLocalRecoWithoutHO)
doLocalCalo = cms.Sequence(doLocalEcal+doLocalHcal)
doLocalCaloWithoutHO = cms.Sequence(doLocalEcal+doLocalHcalWithoutHO)
doRegionalEgammaEcal = cms.Sequence(ecalRegionalEgammaRecoSequence)
doRegionalMuonsEcal = cms.Sequence(ecalRegionalMuonsRecoSequence)
doRegionalTausEcal = cms.Sequence(ecalRegionalTausRecoSequence)
doRegionalJetsEcal = cms.Sequence(ecalRegionalJetsRecoSequence)
doEcalAll = cms.Sequence(ecalAllRecoSequence)
doLocalCSC = cms.Sequence(cms.SequencePlaceholder("muonCSCDigis")*csclocalreco)
doLocalDT = cms.Sequence(cms.SequencePlaceholder("muonDTDigis")*dtlocalreco)
doLocalRPC = cms.Sequence(cms.SequencePlaceholder("muonRPCDigis")*rpcRecHits)
doLocalMuon = cms.Sequence(doLocalDT+doLocalCSC+doLocalRPC)

