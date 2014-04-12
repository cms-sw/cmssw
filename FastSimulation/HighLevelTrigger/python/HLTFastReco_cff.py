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
from FastSimulation.HighLevelTrigger.HLTFastRecoForL1FastJet_cff import *
#from FastSimulation.HighLevelTrigger.HLTFastRecoForPF_cff import *   # IT IS NOT NEEDED ANY MORE IN 44X
from FastSimulation.HighLevelTrigger.HLTFastRecoForXchannel_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForSpecial_cff import *

# L1 emulator - in the future, we may want to use directly L1Trigger.Configuration.SimL1Emulator_cff
# Configuration comes from the GlobalTag
# Emulator modules
from L1Trigger.Configuration.L1MuonEmulator_cff import *
from L1Trigger.Configuration.L1CaloEmulator_cff import *
from L1Trigger.GlobalTrigger.gtDigis_cfi import *
rctDigis.ecalDigis = cms.VInputTag(cms.InputTag("simEcalTriggerPrimitiveDigis"))
rctDigis.hcalDigis = cms.VInputTag(cms.InputTag("simHcalTriggerPrimitiveDigis"))
# Emulator sequence
L1Emulator = cms.Sequence(L1CaloEmulator*L1MuonEmulator*gtDigis)

# The calorimeter emulator requires doDigis=true)
CaloMode = 0   ### In CMSSW > 61X CaloMode can be updated with the following import
from FastSimulation.CaloRecHitsProducer.CaloRecHits_cff import *
if(CaloMode==0 or CaloMode==2):
    ecalRecHit.doDigis = True
if(CaloMode==0 or CaloMode==1):
    hbhereco.doDigis = True
    hfreco.doDigis = True
    horeco.doDigis = True

# L1 muons emulator
#from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi import *
from L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi import *
dtTriggerPrimitiveDigis.digiTag = cms.InputTag("simMuonDTDigis")
from L1Trigger.RPCTrigger.rpcTriggerDigis_cfi import *
rpcTriggerDigis.label = "simMuonRPCDigis"
from L1Trigger.GlobalMuonTrigger.gmtDigis_cfi import *
gmtDigis.DTCandidates = cms.InputTag("dttfDigis","DT")
gmtDigis.CSCCandidates = cms.InputTag("csctfDigis","CSC")
gmtDigis.RPCbCandidates = cms.InputTag("rpcTriggerDigis","RPCb")
gmtDigis.RPCfCandidates = cms.InputTag("rpcTriggerDigis","RPCf")
gmtDigis.MipIsoData = cms.InputTag("rctDigis")

# GT emulator
gtDigis.EmulateBxInEvent = 1


# L1Extra - provides 4-vector representation of L1 trigger objects - not needed by HLT
# The muon extra particles are done here, but could be done also by L1ParamMuons.
from L1Trigger.Configuration.L1Extra_cff import *
l1extraParticles.muonSource = 'gmtDigis'

# L1 report
import L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi
hltL1GtTrigReport = L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi.l1GtTrigReport.clone()
hltL1GtTrigReport.PrintVerbosity = 1
hltL1GtTrigReport.PrintOutput = 2

# HLT Report
options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false
)

# basic tracking stuff
from FastSimulation.TrackingRecHitProducer.SiTrackerGaussianSmearingRecHitConverter_cfi import *
from FastSimulation.Tracking.IterativeTracking_cff import *

# The hltbegin sequence (with L1 emulator)
HLTBeginSequence = cms.Sequence(
    siTrackerGaussianSmearingRecHits+ # repetition if RECO is executed; needed by the next line
    iterativeTracking+ # repetition if RECO is executed; needed by the next line
    caloRecHits+ # repetition if RECO is executed; needed to allow -s GEN,SIM,HLT without RECO
    L1CaloEmulator+
    L1MuonEmulator+
    gtDigis+
    l1extraParticles+
    cms.SequencePlaceholder("offlineBeamSpot")
)

HLTBeginSequenceBPTX = cms.Sequence(HLTBeginSequence)



