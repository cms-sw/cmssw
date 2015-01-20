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

# L1 emulator - using directly L1Trigger.Configuration.SimL1Emulator_cff
# for everithing but simRctDigis that is taken from CaloRecHits_cff
#
# GT digis and L1 extra have different module label naming  w.r.t.
# FullSim as they are used as input to HLT w.o. any packing/unpacking
#
# In general configuration for the emulator modules comes from GlobalTag

from L1Trigger.Configuration.SimL1Emulator_cff import simGctDigis,             \
    simDtTriggerPrimitiveDigis, L1DTConfigFromDB, simCscTriggerPrimitiveDigis, \
    simCsctfTrackDigis, simDttfDigis, simCsctfDigis,                           \
    simRpcTriggerDigis, RPCConeBuilder, simGmtDigis,                           \
    SimL1MuTriggerPrimitives, SimL1MuTrackFinders

from L1Trigger.GlobalTrigger.gtDigis_cfi import *

# The calorimeter emulator requires doDigis=true
from FastSimulation.CaloRecHitsProducer.CaloRecHits_cff import *

# GT emulator
gtDigis.EmulateBxInEvent = 1
gtDigis.GmtInputTag = cms.InputTag("simGmtDigis") 
gtDigis.GctInputTag = cms.InputTag("simGctDigis")

# Emulator sequence
L1Emulator = cms.Sequence(simRctDigis + 
                          simGctDigis + 
                          SimL1MuTriggerPrimitives + 
                          SimL1MuTrackFinders + 
                          simRpcTriggerDigis + 
                          simGmtDigis +
                          gtDigis)

# L1Extra - provides 4-vector representation of L1 trigger objects - not needed by HLT
# The muon extra particles are done here, but could be done also by L1ParamMuons.
from L1Trigger.Configuration.L1Extra_cff import *

l1extraParticles.isolatedEmSource    = cms.InputTag("simGctDigis","isoEm")
l1extraParticles.nonIsolatedEmSource = cms.InputTag("simGctDigis","nonIsoEm")

l1extraParticles.centralJetSource = cms.InputTag("simGctDigis","cenJets")
l1extraParticles.tauJetSource     = cms.InputTag("simGctDigis","tauJets")
l1extraParticles.isoTauJetSource  = cms.InputTag("simGctDigis","isoTauJets")
l1extraParticles.forwardJetSource = cms.InputTag("simGctDigis","forJets")

l1extraParticles.muonSource = cms.InputTag('simGmtDigis')

l1extraParticles.etTotalSource = cms.InputTag("simGctDigis")
l1extraParticles.etHadSource   = cms.InputTag("simGctDigis")
l1extraParticles.htMissSource  = cms.InputTag("simGctDigis")
l1extraParticles.etMissSource  = cms.InputTag("simGctDigis")

l1extraParticles.hfRingEtSumsSource    = cms.InputTag("simGctDigis")
l1extraParticles.hfRingBitCountsSource = cms.InputTag("simGctDigis")


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
    iterativeTracking               + # repetition if RECO is executed; needed by the next line
    caloRecHits                     + # repetition if RECO is executed; needed to allow -s GEN,SIM,HLT without RECO
    L1Emulator                      +
    l1extraParticles                +
    cms.SequencePlaceholder("offlineBeamSpot")
)

HLTBeginSequenceBPTX = cms.Sequence(HLTBeginSequence)



