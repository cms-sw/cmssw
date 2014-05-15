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
hltIter4Tau3MuMerged = HLTgeneralTracks.clone()
hltIter4MergedReg = HLTgeneralTracks.clone()
hltIter2MergedForElectrons = HLTgeneralTracks.clone()
hltIter2MergedForPhotons = HLTgeneralTracks.clone()
hltIter2L3MuonMergedReg = HLTgeneralTracks.clone()
hltIter2GlbTrkMuonMergedReg = HLTgeneralTracks.clone()
hltForBTagIter2Merged = HLTgeneralTracks.clone()

HLTIterativeTracking = cms.Sequence(hltIter4Merged)
HLTIterativeTrackingTau3Mu = cms.Sequence(hltIter4Tau3MuMerged)
HLTIterativeTrackingReg = cms.Sequence(hltIter4MergedReg)
HLTIterativeTrackingForEgamma = cms.Sequence(hltIter2MergedForElectrons)
HLTIterativeTrackingForPhotons = cms.Sequence(hltIter2MergedForPhotons)
HLTIterativeTrackingL3MuonRegIter02 = cms.Sequence(hltIter2L3MuonMergedReg)
HLTIterativeTrackingGlbTrkMuonReg = cms.Sequence(hltIter2GlbTrkMuonMergedReg)
HLTForBTagIterativeTracking = cms.Sequence(hltForBTagIter2Merged)
