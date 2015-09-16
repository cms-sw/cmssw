import FWCore.ParameterSet.Config as cms

### HLT filter
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
WMuHLTFilter = copy.deepcopy(hltHighLevel)
WMuHLTFilter.throw = cms.bool(False)
WMuHLTFilter.HLTPaths = ["HLT_Mu*","HLT_IsoMu*"]

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
pfMetWMuNuSeq = cms.Sequence(WMuHLTFilter *
                             goodMuonsForW *
                             wmnPFCands *
                             wmnPFFilter
                             )


tcMetWMuNuSeq = cms.Sequence(WMuHLTFilter *
                             goodMuonsForW *
                             wmnTCCands *
                             wmnTCFilter
                             )


