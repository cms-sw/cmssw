import FWCore.ParameterSet.Config as cms

### HLT filter
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
WZMuHLTFilter = copy.deepcopy(hltHighLevel)
WZMuHLTFilter.throw = cms.bool(False)
WZMuHLTFilter.HLTPaths = ["HLT_Mu9","HLT_Mu11","HLT_Mu15"]

### Z -> MuMu candidates

# Get muons of needed quality for Zs
goodMuonsForZ = cms.EDFilter("MuonSelector",
                             src = cms.InputTag("muons"),
                             cut = cms.string('pt > 20 && abs(eta)<2.4 && isGlobalMuon = 1 && isTrackerMuon = 1 && isolationR03().sumPt<3.0 && abs(innerTrack().dxy)<1.0'),
                             filter = cms.bool(True)                                
                             )

# build Z-> MuMu candidates
dimuons = cms.EDProducer("CandViewShallowCloneCombiner",
                         checkCharge = cms.bool(True),
                         cut = cms.string('mass > 60'),
                         decay = cms.string("goodMuonsForZ@+ goodMuonsForZ@-")
                         )

# Z filter
dimuonsFilter = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("dimuons"),
                             minNumber = cms.uint32(1)
                             )

# Z Skim sequence
diMuonSelSeq = cms.Sequence(WZMuHLTFilter *
                            goodMuonsForZ *
                            dimuons *
                            dimuonsFilter
                            )

### W -> MuNu candidates

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


