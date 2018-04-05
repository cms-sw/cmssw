import FWCore.ParameterSet.Config as cms
### HLT filter
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi import patGenericParticles
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import patMuons

ZMuHLTFilter = copy.deepcopy(hltHighLevel)
ZMuHLTFilter.throw = cms.bool(False)
ZMuHLTFilter.HLTPaths = ["HLT_Mu*","HLT_IsoMu*"]

### Z -> MuMu candidates
# Get muons of needed quality for Zs

###create a track collection with generic kinematic cuts
looseMuonsForZMuSkim = cms.EDFilter("TrackSelector",
                             src = cms.InputTag("generalTracks"),
                             cut = cms.string('pt > 10 &&  abs(eta)<2.4 &&  (charge!=0)'),
                             filter = cms.bool(True)                                
                             )



###cloning the previous collection into a collection of candidates
ConcretelooseMuonsForZMuSkim = cms.EDProducer("ConcreteChargedCandidateProducer",
                                              src = cms.InputTag("looseMuonsForZMuSkim"),
                                              particleType = cms.string("mu+")
                                              )



###create iso deposits
tkIsoDepositTk = cms.EDProducer("CandIsoDepositProducer",
                                src = cms.InputTag("ConcretelooseMuonsForZMuSkim"),
                                MultipleDepositsFlag = cms.bool(False),
                                trackType = cms.string('track'),
                                ExtractorPSet = cms.PSet(
        #MIsoTrackExtractorBlock
        Diff_z = cms.double(0.2),
        inputTrackCollection = cms.InputTag("generalTracks"),
        BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        ComponentName = cms.string('TrackExtractor'),
        DR_Max = cms.double(0.5),
        Diff_r = cms.double(0.1),
        Chi2Prob_Min = cms.double(-1.0),
        DR_Veto = cms.double(0.01),
        NHits_Min = cms.uint32(0),
        Chi2Ndof_Max = cms.double(1e+64),
        Pt_Min = cms.double(-1.0),
        DepositLabel = cms.untracked.string('tracker'),
        BeamlineOption = cms.string('BeamSpotFromEvent')
        )
                                )

###adding isodeposits to candidate collection
allPatTracks = patGenericParticles.clone(
    src = cms.InputTag("ConcretelooseMuonsForZMuSkim"),
    # isolation configurables
    userIsolation = cms.PSet(
      tracker = cms.PSet(
        veto = cms.double(0.015),
        src = cms.InputTag("tkIsoDepositTk"),
        deltaR = cms.double(0.3),
        #threshold = cms.double(1.5)
      ),
      ),
    isoDeposits = cms.PSet(
        tracker = cms.InputTag("tkIsoDepositTk"),
        ),
    )




###create the "probe collection" of isolated tracks 
looseIsoMuonsForZMuSkim = cms.EDFilter("PATGenericParticleSelector",  
                             src = cms.InputTag("allPatTracks"), 
                             cut = cms.string("(userIsolation('pat::TrackIso')/pt)<0.4"),
                             filter = cms.bool(True)
                             )



###create the "tag collection" of muon candidate, embedding the relevant infos  
tightMuonsCandidateForZMuSkim = patMuons.clone(
    src = cms.InputTag("muons"),
    embedHighLevelSelection = cms.bool(True), 
)

##apply ~tight muon ID 
tightMuonsForZMuSkim = cms.EDFilter("PATMuonSelector",
                                    src = cms.InputTag("tightMuonsCandidateForZMuSkim"),       
                                    cut = cms.string('(pt > 28) &&  (abs(eta)<2.4) && (isPFMuon>0) && (isGlobalMuon = 1) && (globalTrack().normalizedChi2() < 10) && (globalTrack().hitPattern().numberOfValidMuonHits()>0)&& (numberOfMatchedStations() > 1)&& (innerTrack().hitPattern().numberOfValidPixelHits() > 0)&& (innerTrack().hitPattern().trackerLayersWithMeasurement() > 5) && (abs(dB)<0.2)  && ((isolationR03().sumPt/pt)<0.1)'),
                                    filter = cms.bool(True)                                
                                    )


# build Z-> MuMu candidates
dimuonsZMuSkim = cms.EDProducer("CandViewShallowCloneCombiner",
                         checkCharge = cms.bool(False),
                         cut = cms.string('(mass > 60) &&  (charge=0) && (abs(daughter(0).vz - daughter(1).vz) < 0.1)'),
                         decay = cms.string("tightMuonsForZMuSkim looseIsoMuonsForZMuSkim")
                         )                                    


# Z filter
dimuonsFilterZMuSkim = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("dimuonsZMuSkim"),
                             minNumber = cms.uint32(1)
                             )



diMuonSelSeq = cms.Sequence(
                            ZMuHLTFilter *
                            looseMuonsForZMuSkim *
                            ConcretelooseMuonsForZMuSkim *
                            tkIsoDepositTk *
                            allPatTracks *
                            looseIsoMuonsForZMuSkim * 
                            tightMuonsCandidateForZMuSkim *
                            tightMuonsForZMuSkim *
                            dimuonsZMuSkim *
                            dimuonsFilterZMuSkim 
)






