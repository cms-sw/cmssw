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
                     'keep *_slimmedJets_*_*',
     )
)


import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi import patGenericParticles
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import patMuons

MuonPOGSkimHLTFilter = copy.deepcopy(hltHighLevel)
MuonPOGSkimHLTFilter.throw = cms.bool(False)
MuonPOGSkimHLTFilter.HLTPaths = ["HLT_Mu*","HLT_IsoMu*"]


MuonPOGJPsiSkimHLTFilter = copy.deepcopy(hltHighLevel)
MuonPOGJPsiSkimHLTFilter.throw = cms.bool(False)
MuonPOGJPsiSkimHLTFilter.HLTPaths = ["HLT_Mu*_Track*_Jpsi*"]


TAGMUON_CUT = '(pt > 25) &&  (abs(eta)<2.4) && (isPFMuon>0) && (isGlobalMuon = 1) && (globalTrack().normalizedChi2() < 10) && (globalTrack().hitPattern().numberOfValidMuonHits()>0)&& (numberOfMatchedStations() > 1)&& (innerTrack().hitPattern().numberOfValidPixelHits() > 0)&& (innerTrack().hitPattern().trackerLayersWithMeasurement() > 5) &&  (((pfIsolationR04.sumChargedHadronPt + max(0., pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - 0.5 * pfIsolationR04.sumPUPt) ) / pt)<0.2)'
TAGMUON_JPSI_CUT = '(isGlobalMuon || numberOfMatchedStations > 1) && pt > 5'

PROBETRACK_CUT = 'pt > 10 &&  abs(eta)<2.4 &&  (charge!=0)'
PROBETRACK_JPSI_CUT = 'pt > 7 &&  abs(eta)<2.4 &&  (charge!=0)'

DIMUON = 'mass > 40 || ( ?daughter(0).masterClone.isStandAloneMuon?({dg0}.p+{dg1}.p)*({dg0}.p+{dg1}.p)-({dg0}.px+{dg1}.px)*({dg0}.px+{dg1}.px)-({dg0}.py+{dg1}.py)*({dg0}.py+{dg1}.py)-({dg0}.pz+{dg1}.pz)*({dg0}.pz+{dg1}.pz):2000) > 1600'

DIMUON = DIMUON.format(dg0 = "daughter(0).masterClone.standAloneMuon()", dg1="daughter(1)")


# Tag and probe for Z#to#mu#mu
GoodTagMuons = cms.EDFilter("MuonRefPatSelector",
                            src = cms.InputTag("slimmedMuons"),
                            cut = cms.string(TAGMUON_CUT)
                            )

GoodProbeMuons = cms.EDFilter("IsoTrackSelector",
                             src = cms.InputTag("isolatedTracks"),
                             cut = cms.string(PROBETRACK_CUT),
                             filter = cms.bool(True)                                
)

# Tag and probe for J/Psi #to #mu#mu
GoodJPsiTagMuons = cms.EDFilter("MuonRefPatSelector",
                                src = cms.InputTag("slimmedMuons"),
                                cut = cms.string(TAGMUON_JPSI_CUT)
                                )

GoodJPsiProbeMuons = cms.EDFilter("IsoTrackSelector",
                                  src = cms.InputTag("isolatedTracks"),
                                  cut = cms.string(PROBETRACK_JPSI_CUT),
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

MuonPOGJPsiSkimSequence = cms.Sequence(
    MuonPOGJPsiSkimHLTFilter *
    GoodJPsiTagMuons * 
    GoodJPsiProbeMuons 
    )
