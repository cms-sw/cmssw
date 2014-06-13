import FWCore.ParameterSet.Config as cms

#from FastSimulation.Tracking.PixelTracksProducer_cff import *
#from FastSimulation.Tracking.PixelVerticesProducer_cff import *
#from FastSimulation.Tracking.IterativeFirstTracking_cff import *
#from FastSimulation.Tracking.IterativeSecondTracking_cff import *
#from FastSimulation.Tracking.IterativeThirdTracking_cff import *
#from FastSimulation.Tracking.IterativeFourthTracking_cff import *
####from FastSimulation.Tracking.IterativeFifthTracking_cff import *
from FastSimulation.Tracking.HLTGeneralTracks_cfi import *
#from TrackingTools.TrackFitters.TrackFitters_cff import *
#from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *

hltIter4Merged = HLTgeneralTracks.clone()
hltIter2Merged = HLTgeneralTracks.clone()
hltIter4Tau3MuMerged = HLTgeneralTracks.clone()
hltIter4MergedReg = HLTgeneralTracks.clone()
hltIter2MergedForElectrons = HLTgeneralTracks.clone()
hltIter2MergedForPhotons = HLTgeneralTracks.clone()
hltIter2L3MuonMerged = HLTgeneralTracks.clone()
hltIter2L3MuonMergedReg = HLTgeneralTracks.clone()
hltIter2MergedForBTag = HLTgeneralTracks.clone()
hltIter4MergedForTau = HLTgeneralTracks.clone()
hltIter2GlbTrkMuonMerged = HLTgeneralTracks.clone()
hltIter2HighPtTkMuMerged  = HLTgeneralTracks.clone()
hltIter2HighPtTkMuIsoMerged  = HLTgeneralTracks.clone()

HLTIterativeTracking = cms.Sequence(hltIter4Merged)
HLTReducedIterativeTracking = cms.Sequence(hltIter2Merged)
HLTIterativeTrackingTau3Mu = cms.Sequence(hltIter4Tau3MuMerged)
HLTIterativeTrackingReg = cms.Sequence(hltIter4MergedReg)
HLTIterativeTrackingForElectronIter02 = cms.Sequence(hltIter2MergedForElectrons)
HLTIterativeTrackingForPhotonsIter02 = cms.Sequence(hltIter2MergedForPhotons)
HLTIterativeTrackingL3MuonIter02 = cms.Sequence(hltIter2L3MuonMerged)
HLTIterativeTrackingL3MuonRegIter02 = cms.Sequence(hltIter2L3MuonMergedReg)
HLTIterativeTrackingForBTagIter02 = cms.Sequence(hltIter2MergedForBTag)
HLTIterativeTrackingForTauIter04  = cms.Sequence(hltIter4MergedForTau)
HLTIterativeTrackingGlbTrkMuonIter02 = cms.Sequence(hltIter2GlbTrkMuonMerged)
HLTIterativeTrackingHighPtTkMu = cms.Sequence(hltIter2HighPtTkMuMerged)
HLTIterativeTrackingHighPtTkMuIsoIter02 = cms.Sequence(hltIter2HighPtTkMuIsoMerged)
