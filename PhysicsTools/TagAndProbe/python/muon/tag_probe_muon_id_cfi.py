import FWCore.ParameterSet.Config as cms

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *

staMuons = cms.EDFilter( "MuonRefSelector", 
                        src = cms.InputTag( "muons" ),
                        cut = cms.string( "pt > 20.0" ))

globalMuons = cms.EDFilter( "MuonRefSelector", 
                           src = cms.InputTag( "muons" ),
                           cut = cms.string( "isGlobalMuon > 0 & pt > 20.0" ))

allTracks = cms.EDFilter( "TrackViewCandidateProducer", 
                          src = cms.InputTag( "generalTracks" ),
                          particleType = cms.string( "mu+" ),
                          cut = cms.string ("pt > 0" ))

tkStaMap = cms.EDFilter( "TrivialDeltaRViewMatcher",
                         src     = cms.InputTag( "allTracks" ),
                         matched = cms.InputTag( "staMuons" ),
                         distMin = cms.double(0.3) )

allTkCands = cms.EDFilter( "RecoChargedCandidateRefSelector",
                           src = cms.InputTag( "allTracks"),
                           cut = cms.string( "pt > 10.0"))

tkStaMatched = cms.EDFilter( "RecoChargedCandidateMatchedProbeMaker", 
                             CandidateSource   = cms.InputTag( "allTkCands" ),
                             ResMatchMapSource = cms.InputTag( "tkStaMap" ),
                             ReferenceSource   = cms.InputTag( "staMuons" ),
                             Matched = cms.bool(True))

globalTkMap = cms.EDFilter( "TrivialDeltaRViewMatcher", 
	src     = cms.InputTag( "tkStaMatched" ),
	matched = cms.InputTag( "globalMuons" ),
	distMin = cms.double(0.3))

globalTkStaMatched = cms.EDFilter( "RecoChargedCandidateMatchedProbeMaker", 
                                   CandidateSource   = src.InputTag( "tkStaMatched" ),
                                   ResMatchMapSource = src.InputTag( "globalTkMap" ),
                                   ReferenceSource   = src.InputTag( "globalMuons" ),
                                   Matched = src.bool(True))

muonTagProbeMap = cms.EDProducer( "TagProbeProducer", 
                                  TagCollection   = cms.InputTag( "globalMuons" ),
                                  ProbeCollection = cms.InputTage( "tkStaMatched" ),
                                  MassMinCut = cms.double(50.0),
                                  MassMaxCut = cms.double(120.0))

tagMuonMatch = cms.EDFilter( "MCTruthDeltaRMatcherNew", 
                          src = cms.InputTag( "globalMuons" ),
                          matched = cms.InputTag( "genParticles" ),
                          distMin = cms.double(0.25),
                          pdgId = cms.vint32(13))


allProbeMuonMatch = cms.EDFilter("MCTruthDeltaRMatcherNew", 
                                 src = cms.InputTag("tkStaMatched"),
                                 matched = cms.InputTag("genParticles"),
                                 distMin = cms.double(0.25),
                                 pdgId = cms.vint32(13))
    
passProbeMuonMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
                                  src = cms.InputTag("globalTkStaMatched"),
                                  matched = cms.InputTag("genParticles"),
                                  distMin = cms.double(0.25),
                                  pdgId =  cms.vint32(13))
  
muon_cands = cms.Sequence(allTracks+allTkCands+staMuons+globalMuons*tkStaMap*tkStaMatched+globalTkMap*globalTkStaMatched+muonTagProbeMap+tagMuonMatch+allProbeMuonMatch+passProbeMuonMatch) 
