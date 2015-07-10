import FWCore.ParameterSet.Config as cms

### HLT filter
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ZMuHLTFilter = copy.deepcopy(hltHighLevel)
ZMuHLTFilter.throw = cms.bool(False)
ZMuHLTFilter.HLTPaths = ["HLT_Mu*","HLT_IsoMu*","HLT_DoubleMu*"]

### Z -> MuMu candidates

# Get muons of needed quality for Zs
looseMuonsForZMuSkim = cms.EDFilter("MuonSelector",
                             src = cms.InputTag("muons"),
                             cut = cms.string('pt > 10 && abs(eta)<2.4 && isGlobalMuon = 1 && isTrackerMuon = 1 && abs(innerTrack().dxy)<2.0'),
                             filter = cms.bool(True)                                
                             )

tightMuonsForZMuSkim = cms.EDFilter("MuonSelector",
                             src = cms.InputTag("looseMuonsForZMuSkim"),
                             cut = cms.string('pt > 20'),
                             filter = cms.bool(True)                                
                             )

# build Z-> MuMu candidates
dimuonsZMuSkim = cms.EDProducer("CandViewShallowCloneCombiner",
                         checkCharge = cms.bool(False),
                         cut = cms.string('mass > 30'),
                         decay = cms.string("tightMuonsForZMuSkim looseMuonsForZMuSkim")
                         )

# Z filter
dimuonsFilterZMuSkim = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("dimuonsZMuSkim"),
                             minNumber = cms.uint32(1)
                             )

# Z Skim sequence
diMuonSelSeq = cms.Sequence(ZMuHLTFilter *
                            looseMuonsForZMuSkim *
                            tightMuonsForZMuSkim *
                            dimuonsZMuSkim *
                            dimuonsFilterZMuSkim
                            )
