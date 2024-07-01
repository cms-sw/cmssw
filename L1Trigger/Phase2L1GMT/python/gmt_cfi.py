from L1Trigger.Phase2L1GMT.gmtStubs_cfi import *
from L1Trigger.Phase2L1GMT.gmtKMTFMuons_cfi import *
from L1Trigger.Phase2L1GMT.gmtFwdMuons_cfi import *
from L1Trigger.Phase2L1GMT.gmtSAMuons_cfi import *
from L1Trigger.Phase2L1GMT.gmtTkMuons_cfi import *
l1tGMTStubs = cms.Sequence(gmtStubs)
l1tGMTMuons = cms.Sequence(gmtKMTFMuons*gmtFwdMuons*gmtSAMuons*gmtTkMuons)

l1tGMTFilteredMuons = cms.EDProducer('Phase2L1TGMTFilter',
                    srcMuons = cms.InputTag("l1tTkMuonsGmt",""),
                    applyLowPtFilter = cms.bool(True),
                    ptBarrelMin = cms.int32(8),
                    ptEndcapMin = cms.int32(8),
                    etaBE = cms.double(0.9)
                                     
)
