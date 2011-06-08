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
                             cut = cms.string('isGlobalMuon=1 && isTrackerMuon=1 && abs(eta)<2.1 && abs(globalTrack().dxy)<0.2 && pt>20. && globalTrack().normalizedChi2<10 && globalTrack().hitPattern().numberOfValidTrackerHits>10 && globalTrack().hitPattern().numberOfValidMuonHits>0 && globalTrack().hitPattern().numberOfValidPixelHits>0 && numberOfMatches>1 && (isolationR03().sumPt+isolationR03().emEt+isolationR03().hadEt)<0.15*pt'),
                             filter = cms.bool(True)
                             )

# build W->MuNu candidates using PF MET
wmnPFCands = cms.EDProducer("CandViewShallowCloneCombiner",
                            checkCharge = cms.bool(False),
                            cut = cms.string('sqrt((daughter(0).pt+daughter(1).pt)*(daughter(0).pt+daughter(1).pt)-pt*pt)>50'),
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
                            cut = cms.string('sqrt((daughter(0).pt+daughter(1).pt)*(daughter(0).pt+daughter(1).pt)-pt*pt)>50'),
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


