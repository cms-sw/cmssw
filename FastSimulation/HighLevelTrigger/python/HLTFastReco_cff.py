import FWCore.ParameterSet.Config as cms

#
# Load subdetector specific common files
#
from FastSimulation.Tracking.PixelVerticesProducer_cff import *
from FastSimulation.Tracking.GlobalPixelTracking_cff import *

#Specific reconstruction sequences for FastSimulation.
from FastSimulation.HighLevelTrigger.HLTFastRecoForJetMET_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForEgamma_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForMuon_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForTau_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForB_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForXchannel_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForSpecial_cff import *

# Muon parametrization (L1 and L3 are produced by HLT paths)
from FastSimulation.ParamL3MuonProducer.ParamL3Muon_cfi import *
paramMuons.MUONS.ProduceL1Muons = False
paramMuons.MUONS.ProduceL3Muons = False

# L1 emulator 
from L1Trigger.Configuration.L1Emulator_cff import *
rctDigis.ecalDigis = cms.VInputTag(cms.InputTag("simEcalTriggerPrimitiveDigis"))
rctDigis.hcalDigis = cms.VInputTag(cms.InputTag("simHcalTriggerPrimitiveDigis"))

# The calorimeter emulator requires doDigis=true)
from FastSimulation.CaloRecHitsProducer.CaloRecHits_cff import *
ecalRecHit.doDigis = True
hbhereco.doDigis = True
hfreco.doDigis = True
horeco.doDigis = True

# The parameterized L1 Muons (much faster than the L1 emulator)
from FastSimulation.Muons.L1Muons_cfi import *

# GT emulator
# the replace of the DaqActiveBoard allows the GT to run without muon inputs (no longer needed)
gtDigis.GmtInputTag = 'l1ParamMuons'
gtDigis.EmulateBxInEvent = 1


# L1Extra - provides 4-vector representation of L1 trigger objects - not needed by HLT
# The muon extra particles are done by L1ParamMuons, but could be done here too.
# Need to check the difference of efficiencies first.
from L1Trigger.Configuration.L1Extra_cff import *
l1extraParticles.muonSource = 'l1ParamMuons'

# L1 report
import L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi
hltL1gtTrigReport = L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi.l1GtTrigReport.clone()
hltL1gtTrigReport.PrintVerbosity = 1
hltL1gtTrigReport.PrintOutput = 2

# HLT Report
options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false
)

# The hltbegin sequence (with L1 emulator)
HLTBeginSequence = cms.Sequence(
    cms.SequencePlaceholder("simulation")+
    cms.SequencePlaceholder("simEcalTriggerPrimitiveDigis")+
    simHcalTriggerPrimitiveDigis+
    L1CaloEmulator+
    l1ParamMuons+
    gtDigis+
    l1extraParticles+
    cms.SequencePlaceholder("offlineBeamSpot")
)

# An older L1 sequence (with L1 simulator)
# this one cannot be used by the HLT as of 17X  use the previous sequence instead 
# Fast L1 Trigger
#from FastSimulation.L1CaloTriggerProducer.fastl1calosim_cfi import *
#from FastSimulation.L1CaloTriggerProducer.fastL1extraParticleMap_cfi import *
#fastL1CaloSim.AlgorithmSource = 'RecHits'
#fastL1CaloSim.EmInputs = cms.VInputTag(
#    cms.InputTag("caloRecHits","EcalRecHitsEB"),
#    cms.InputTag("caloRecHits","EcalRecHitsEE")
#)
#famosWithL1 = cms.Sequence(
#    cms.SequencePlaceholder("famosWithCaloTowers")+
#    cms.SequencePlaceholder("simEcalTriggerPrimitiveDigis")+
#    simHcalTriggerPrimitiveDigis+fastL1CaloSim+
#    fastL1extraParticleMap
#)


