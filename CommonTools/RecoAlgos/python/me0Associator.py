import FWCore.ParameterSet.Config as cms


#----------ME0Muon Collection Production for association by chi2
me0muon = cms.EDProducer("ME0MuonTrackCollProducer",
                         me0MuonTag = cms.InputTag("me0SegmentMatching"),
                         selectionTags = cms.vstring('All'),
                         )
#--------------------
me0muonColl_seq = cms.Sequence(
                             me0muon
                             )
