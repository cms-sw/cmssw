import FWCore.ParameterSet.Config as cms

# Pick branches you want to keep
MuonPOG_EventContent = cms.PSet(
     outputCommands = cms.untracked.vstring(
                     'drop *',
                     'keep *_TriggerResults_*_*',
                     'keep *_gmtStage2Digis_Muon_*',
                     'keep *_offlineSlimmedPrimaryVertices_*_*',
                     'keep *_offlineBeamSpot_*_*',                     
                     'keep *_slimmedMETs_*_*',
                     'keep *Muons*_slimmedMuons_*_*',
                     'keep *_isolatedTracks_*_*',                   
                     'keep *_slimmedPatTrigger_*_*',
     )
)


import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi import patGenericParticles
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import patMuons

MuonPOGSkimHLTFilter = copy.deepcopy(hltHighLevel)
MuonPOGSkimHLTFilter.throw = cms.bool(False)
MuonPOGSkimHLTFilter.HLTPaths = ["HLT_Mu*","HLT_IsoMu*"]

TAGMUON_CUT = '(pt > 28) &&  (abs(eta)<2.4) && (isPFMuon>0) && (isGlobalMuon = 1) && (globalTrack().normalizedChi2() < 10) && (globalTrack().hitPattern().numberOfValidMuonHits()>0)&& (numberOfMatchedStations() > 1)&& (innerTrack().hitPattern().numberOfValidPixelHits() > 0)&& (innerTrack().hitPattern().trackerLayersWithMeasurement() > 5) &&  ((isolationR03().sumPt/pt)<0.1)'
PROBETRACK_CUT = 'pt > 10 &&  abs(eta)<2.4 &&  (charge!=0)'
DIMUON = 'mass > 40'

GoodTagMuons = cms.EDFilter("MuonRefPatSelector",
                            src = cms.InputTag("slimmedMuons"),
                            cut = cms.string(TAGMUON_CUT)
                            )

GoodProbeMuons = cms.EDFilter("IsoTrackSelector",
                             src = cms.InputTag("isolatedTracks"),
                             cut = cms.string(PROBETRACK_CUT),
                             filter = cms.bool(True)                                
)

# build Z-> MuMu candidates
DiMuonPOGSkim = cms.EDProducer("CandViewShallowCloneCombiner",
                                checkCharge = cms.bool(False),
                                cut = cms.string(DIMUON),
                                decay = cms.string("GoodTagMuons GoodProbeMuons")
                                )                                    

# Z filter
DiMuonPOGFilterSkim = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("DiMuonPOGSkim"),
                             minNumber = cms.uint32(1)
                             )


MuonPOGSkimSequence = cms.Sequence(
    MuonPOGSkimHLTFilter *
    GoodTagMuons * 
    GoodProbeMuons *
    DiMuonPOGSkim *
    DiMuonPOGFilterSkim
    )
