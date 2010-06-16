import FWCore.ParameterSet.Config as cms

#from HLTrigger.HLTfilters.hltHighLevel_cfi import *
#exoticaMuHLT = hltHighLevel
#Define the HLT path to be used.
#exoticaMuHLT.HLTPaths =['HLT_L1MuOpen']
#exoticaMuHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")

#Define the HLT quality cut 
#exoticaHLTMuonFilter = cms.EDFilter("HLTSummaryFilter",
#    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
#    member  = cms.InputTag("hltL3MuonCandidates","","HLT8E29"),      # filter or collection									
#    cut     = cms.string("pt>0"),                     # cut on trigger object
#    minN    = cms.int32(0)                  # min. # of passing objects needed
# )
                               

#Define the Reco quality cut
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# Make the charged candidate collections from tracks
allTracks = cms.EDProducer("TrackViewCandidateProducer",
                               src = cms.InputTag("generalTracks"),
                               particleType = cms.string('mu+'),
                               cut = cms.string('pt > 0'),
                           filter = cms.bool(True)
                           )

staTracks = cms.EDProducer("TrackViewCandidateProducer",
                               src = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
                               particleType = cms.string('mu+'),
                               cut = cms.string('pt > 0.5 && abs(d0) < 2.0 && abs(vz) < 25.0'),
                           filter = cms.bool(True)
                           )

# Make the input candidate collections
tagCands = cms.EDFilter("MuonRefSelector",
                            src = cms.InputTag("muons"),
                            cut = cms.string('isGlobalMuon > 0 && pt > 1.0 && abs(eta) < 2.1'),
                        filter = cms.bool(True)
                        )

# Standalone muon tracks (probes)
staCands = cms.EDFilter("RecoChargedCandidateRefSelector",
                            src = cms.InputTag("staTracks"),
                            cut = cms.string('pt > 0.5 && abs(eta) < 2.1'),
                        filter = cms.bool(True)
                        )

# Tracker muons (to be matched)
tkProbeCands = cms.EDFilter("RecoChargedCandidateRefSelector",
                                src = cms.InputTag("allTracks"),
                                cut = cms.string('pt > 0.5'),
                            filter = cms.bool(True)
                            )

# Match track and stand alone candidates
# to get the passing probe candidates
TkStaMap = cms.EDProducer("TrivialDeltaRViewMatcher",
                            src = cms.InputTag("tkProbeCands"),
                            distMin = cms.double(0.15),
                            matched = cms.InputTag("staCands"),
                        filter = cms.bool(True)
                        )

# Use the producer to get a list of matched candidates
TkStaMatched = cms.EDProducer("RecoChargedCandidateMatchedProbeMaker",
                                Matched = cms.untracked.bool(True),
                                ReferenceSource = cms.untracked.InputTag("staCands"),
                                ResMatchMapSource = cms.untracked.InputTag("TkStaMap"),
                                CandidateSource = cms.untracked.InputTag("tkProbeCands"),
                            filter = cms.bool(True)
                            )

TkStaUnmatched = cms.EDProducer("RecoChargedCandidateMatchedProbeMaker",
                                  Matched = cms.untracked.bool(False),
                                  ReferenceSource = cms.untracked.InputTag("staCands"),
                                  ResMatchMapSource = cms.untracked.InputTag("TkStaMap"),
                                  CandidateSource = cms.untracked.InputTag("tkProbeCands"),
                              filter = cms.bool(True)
                              )

# Make the tag probe association map
JPsiMMTagProbeMap = cms.EDProducer("TagProbeMassProducer",
                                     MassMaxCut = cms.untracked.double(4.5),
                                     TagCollection = cms.InputTag("tagCands"),
                                     MassMinCut = cms.untracked.double(1.5),
                                     ProbeCollection = cms.InputTag("tkProbeCands"),
                                     PassingProbeCollection = cms.InputTag("TkStaMatched")
                                 )

JPsiMMTPFilter = cms.EDFilter("TagProbeMassEDMFilter",
                        tpMapName = cms.string('JPsiMMTagProbeMap')
                        )

ZMMTagProbeMap = cms.EDProducer("TagProbeMassProducer",
                                 MassMaxCut = cms.untracked.double(120.0),
                                 TagCollection = cms.InputTag("tagCands"),
                                 MassMinCut = cms.untracked.double(50.0),
                                 ProbeCollection = cms.InputTag("tkProbeCands"),
                                 PassingProbeCollection = cms.InputTag("TkStaMatched")
                                 )

ZMMTPFilter = cms.EDFilter("TagProbeMassEDMFilter",
                        tpMapName = cms.string('ZMMTagProbeMap')
                        )


#Define group sequence, using HLT/Reco quality cut. 
#exoticaMuHLTQualitySeq = cms.Sequence()
tagProbeSeq = cms.Sequence(allTracks+staTracks*tagCands+tkProbeCands+staCands*TkStaMap*TkStaMatched)

muonJPsiMMRecoQualitySeq = cms.Sequence(
    #exoticaMuHLT+
    tagProbeSeq+JPsiMMTagProbeMap+JPsiMMTPFilter
    )

muonZMMRecoQualitySeq = cms.Sequence(
    #exoticaMuHLT+
    tagProbeSeq+ZMMTagProbeMap+ZMMTPFilter
    )

