import FWCore.ParameterSet.Config as cms

### HLT filter
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
WZMuHLTFilter = copy.deepcopy(hltHighLevel)
WZMuHLTFilter.throw = cms.bool(False)
WZMuHLTFilter.HLTPaths = ["HLT_Mu9","HLT_Mu11","HLT_Mu15","HLT_Mu15_v*"]

### Z -> MuMu candidates

# Get muons of needed quality for Zs
looseMuonsForZ = cms.EDFilter("MuonSelector",
                             src = cms.InputTag("muons"),
                             cut = cms.string('obj.pt() > 10 && std::abs(obj.eta())<2.4 && obj.isGlobalMuon() == 1 && obj.isTrackerMuon() == 1 && std::abs(obj.innerTrack()->dxy())<2.0'),
                             filter = cms.bool(True)                                
                             )

tightMuonsForZ = cms.EDFilter("MuonSelector",
                             src = cms.InputTag("looseMuonsForZ"),
                             cut = cms.string('obj.pt() > 20'),
                             filter = cms.bool(True)                                
                             )

# build Z-> MuMu candidates
dimuons = cms.EDProducer("CandViewShallowCloneCombiner",
                         checkCharge = cms.bool(False),
                         cut = cms.string('obj.mass() > 30'),
                         decay = cms.string("tightMuonsForZ looseMuonsForZ")
                         )

# Z filter
dimuonsFilter = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("dimuons"),
                             minNumber = cms.uint32(1)
                             )

# Z filter
dimuonsFilter = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("dimuons"),
                             minNumber = cms.uint32(1)
                             )

# Z Skim sequence
diMuonSelSeq = cms.Sequence(WZMuHLTFilter *
                            looseMuonsForZ *
                            tightMuonsForZ *
                            dimuons *
                            dimuonsFilter
                            )

### W -> MuNu candidates

#Get muons of needed quality for Ws
goodMuonsForW = cms.EDFilter("MuonViewRefSelector",
                             src = cms.InputTag("muons"),
                             cut = cms.string('obj.isGlobalMuon()==1 && obj.isTrackerMuon()==1 && std::abs(obj.eta())<2.1 && std::abs(obj.globalTrack()->dxy())<0.2 && obj.pt()>20. && obj.globalTrack()->normalizedChi2()<10 && obj.globalTrack()->hitPattern().numberOfValidTrackerHits()>10 && obj.globalTrack()->hitPattern().numberOfValidMuonHits()>0 && obj.globalTrack()->hitPattern().numberOfValidPixelHits()>0 && obj.numberOfMatches()>1 && (obj.isolationR03().sumPt+obj.isolationR03().emEt+obj.isolationR03().hadEt)<0.15*obj.pt()'),
                             filter = cms.bool(True)
                             )

# build W->MuNu candidates using PF MET
wmnPFCands = cms.EDProducer("CandViewShallowCloneCombiner",
                            checkCharge = cms.bool(False),
                            cut = cms.string('std::sqrt((obj.daughter(0)->pt()+obj.daughter(1)->pt())*(obj.daughter(0)->pt()+obj.daughter(1)->pt())-obj.pt()*obj.pt())>50'),
                            decay = cms.string("goodMuonsForW pfMet")
                            )

# W filter
wmnPFFilter = cms.EDFilter("CandViewCountFilter",
                           src = cms.InputTag("wmnPFCands"),
                           minNumber = cms.uint32(1)
                           )

# build W->MuNu candidates using TC MET
wmnTCCands = cms.EDProducer("CandViewShallowCloneCombiner",
                            checkCharge = cms.bool(False),
                            cut = cms.string('std::sqrt((obj.daughter(0)->pt()+obj.daughter(1)->pt())*(obj.daughter(0)->pt()+obj.daughter(1)->pt())-obj.pt()*obj.pt())>50'),
                            decay = cms.string("goodMuonsForW tcMet")
                            )

# W filter
wmnTCFilter = cms.EDFilter("CandViewCountFilter",
                           src = cms.InputTag("wmnTCCands"),
                           minNumber = cms.uint32(1)
                           )


# W Skim sequences
pfMetWMuNuSeq = cms.Sequence(WZMuHLTFilter *
                             goodMuonsForW *
                             wmnPFCands *
                             wmnPFFilter
                             )


tcMetWMuNuSeq = cms.Sequence(WZMuHLTFilter *
                             goodMuonsForW *
                             wmnTCCands *
                             wmnTCFilter
                             )


