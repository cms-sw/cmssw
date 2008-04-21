import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.common.CaloTowers_cff import *
from FastSimulation.Tracking.PixelVerticesProducer_cff import *
from FastSimulation.Tracking.GlobalPixelTracking_cff import *
from FastSimulation.HighLevelTrigger.JetMET.HLTFastRecoForJetMET_cff import *
from FastSimulation.HighLevelTrigger.Egamma.HLTFastRecoForEgamma_cff import *
from FastSimulation.HighLevelTrigger.Muon.HLTFastRecoForMuon_cff import *
from FastSimulation.HighLevelTrigger.btau.HLTFastRecoForTau_cff import *
from FastSimulation.HighLevelTrigger.btau.HLTFastRecoForB_cff import *
from FastSimulation.HighLevelTrigger.xchannel.HLTFastRecoForXchannel_cff import *
from FastSimulation.HighLevelTrigger.special.HLTFastRecoForSpecial_cff import *
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
from FastSimulation.ParamL3MuonProducer.ParamL3Muon_cfi import *
from L1Trigger.Configuration.L1Emulator_cff import *
from FastSimulation.Muons.L1Muons_cfi import *
from L1Trigger.Configuration.L1Extra_cff import *
from FastSimulation.L1CaloTriggerProducer.fastl1calosim_cfi import *
from FastSimulation.L1CaloTriggerProducer.fastL1extraParticleMap_cfi import *
hltBegin = cms.Sequence(cms.SequencePlaceholder("simulation")+cms.SequencePlaceholder("ecalTriggerPrimitiveDigis")+hcalTriggerPrimitiveDigis+L1CaloEmulator+l1ParamMuons+gtDigis+l1extraParticles+cms.SequencePlaceholder("offlineBeamSpot"))
famosWithL1 = cms.Sequence(cms.SequencePlaceholder("famosWithCaloTowers")+cms.SequencePlaceholder("ecalTriggerPrimitiveDigis")+hcalTriggerPrimitiveDigis+fastL1CaloSim+fastL1extraParticleMap)
towerMaker.ecalInputs = cms.VInputTag(cms.InputTag("caloRecHits","EcalRecHitsEB"), cms.InputTag("caloRecHits","EcalRecHitsEE"))
towerMaker.hbheInput = 'caloRecHits'
towerMaker.hfInput = 'caloRecHits'
towerMaker.hoInput = 'caloRecHits'
paramMuons.MUONS.ProduceL1Muons = False
paramMuons.MUONS.ProduceL3Muons = False
caloRecHits.RecHitsFactory.doDigis = True
gtDigis.GmtInputTag = 'l1ParamMuons'
l1extraParticles.muonSource = 'l1ParamMuons'
l1gtTrigReport.PrintVerbosity = 1
l1gtTrigReport.PrintOutput = 2
fastL1CaloSim.AlgorithmSource = 'RecHits'
fastL1CaloSim.EmInputs = cms.VInputTag(cms.InputTag("caloRecHits","EcalRecHitsEB"), cms.InputTag("caloRecHits","EcalRecHitsEE"))

