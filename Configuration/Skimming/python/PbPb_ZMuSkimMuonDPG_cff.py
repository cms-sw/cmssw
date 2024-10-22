import FWCore.ParameterSet.Config as cms
### HLT filter
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi import patGenericParticles


PbPbZMuHLTFilter = copy.deepcopy(hltHighLevel)
PbPbZMuHLTFilter.throw = cms.bool(False)
PbPbZMuHLTFilter.HLTPaths = ["HLT_HIL3Mu*"]

### Z -> MuMu candidates
# Get muons of needed quality for Zs

###create a track collection with generic kinematic cuts
looseMuonsForPbPbZMuSkim = cms.EDFilter("TrackSelector",
                             src = cms.InputTag("generalTracks"),
                             cut = cms.string('pt > 10 &&  abs(eta)<2.4 &&  (charge!=0)'),
                             filter = cms.bool(True)                                
                             )



###cloning the previous collection into a collection of candidates
ConcretelooseMuonsForPbPbZMuSkim = cms.EDProducer("ConcreteChargedCandidateProducer",
                                              src = cms.InputTag("looseMuonsForPbPbZMuSkim"),
                                              particleType = cms.string("mu+")
                                              )



###create iso deposits
tkIsoDepositTkForPbPbZMuSkim = cms.EDProducer("CandIsoDepositProducer",
                                src = cms.InputTag("ConcretelooseMuonsForPbPbZMuSkim"),
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
allPatTracksForPbPbZMuSkim = patGenericParticles.clone(
    src = cms.InputTag("ConcretelooseMuonsForPbPbZMuSkim"),
    # isolation configurables
    userIsolation = cms.PSet(
      tracker = cms.PSet(
        veto = cms.double(0.015),
        src = cms.InputTag("tkIsoDepositTkForPbPbZMuSkim"),
        deltaR = cms.double(0.3),
        #threshold = cms.double(1.5)
      ),
      ),
    isoDeposits = cms.PSet(
        tracker = cms.InputTag("tkIsoDepositTkForPbPbZMuSkim"),
        ),
    )




###create the "probe collection" of isolated tracks 
looseIsoMuonsForPbPbZMuSkim = cms.EDFilter("PATGenericParticleSelector",  
                             src = cms.InputTag("allPatTracksForPbPbZMuSkim"), 
                             cut = cms.string("(userIsolation('pat::TrackIso')/pt)<0.4"),
                             filter = cms.bool(True)
                             )



###create the "tag collection" of muon candidate, no dB cut applied 


tightMuonsForPbPbZMuSkim = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("(isGlobalMuon) && pt > 25. &&  (abs(eta)<2.4) &&  (isPFMuon>0) && (globalTrack().normalizedChi2() < 10) && (globalTrack().hitPattern().numberOfValidMuonHits()>0)&& (numberOfMatchedStations() > 1)&& (innerTrack().hitPattern().numberOfValidPixelHits() > 0)&& (innerTrack().hitPattern().trackerLayersWithMeasurement() > 5) && ((isolationR03().sumPt/pt)<0.1)"),
    filter = cms.bool(True)
    )




# build Z-> MuMu candidates
dimuonsForPbPbZMuSkim = cms.EDProducer("CandViewShallowCloneCombiner",
                         checkCharge = cms.bool(False),
                         cut = cms.string('(mass > 60) &&  (charge=0)'),       
                         decay = cms.string("tightMuonsForPbPbZMuSkim looseIsoMuonsForPbPbZMuSkim")
                         )                                    


# Z filter
dimuonsFilterForPbPbZMuSkim = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("dimuonsForPbPbZMuSkim"),
                             minNumber = cms.uint32(1)
                             )



diMuonSelSeqForPbPbZMuSkim = cms.Sequence(
                            PbPbZMuHLTFilter *
                            looseMuonsForPbPbZMuSkim *
                            ConcretelooseMuonsForPbPbZMuSkim *
                            tkIsoDepositTkForPbPbZMuSkim *
                            allPatTracksForPbPbZMuSkim *
                            looseIsoMuonsForPbPbZMuSkim * 
                            tightMuonsForPbPbZMuSkim *
                            dimuonsForPbPbZMuSkim *
                            dimuonsFilterForPbPbZMuSkim 
)






