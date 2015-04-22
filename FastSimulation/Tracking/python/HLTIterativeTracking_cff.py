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
hltIter4HighPtMerged = HLTgeneralTracks.clone()
hltIter2HighPtMerged = HLTgeneralTracks.clone()
hltIter4Tau3MuMerged = HLTgeneralTracks.clone()
hltIter4MergedReg = HLTgeneralTracks.clone()
hltIter2MergedForElectrons = HLTgeneralTracks.clone()
hltIter2MergedForPhotons = HLTgeneralTracks.clone()
hltIter2L3MuonMerged = HLTgeneralTracks.clone()
hltIter2L3MuonMergedReg = HLTgeneralTracks.clone()
hltIter2MergedForBTag = HLTgeneralTracks.clone()
hltIter2MergedForTau = HLTgeneralTracks.clone()
hltIter4MergedForTau = HLTgeneralTracks.clone()
hltIter2GlbTrkMuonMerged = HLTgeneralTracks.clone()
hltIter2HighPtTkMuMerged  = HLTgeneralTracks.clone()
hltIter2HighPtTkMuIsoMerged  = HLTgeneralTracks.clone()
hltIter2DisplacedJpsiMerged     = HLTgeneralTracks.clone()
hltIter2DisplacedPsiPrimeMerged = HLTgeneralTracks.clone()
hltIter2DisplacedNRMuMuMerged   = HLTgeneralTracks.clone()
hltIter0PFlowTrackSelectionHighPurityForBTag = HLTgeneralTracks.clone()
hltIter4MergedWithIter012DisplacedJets = HLTgeneralTracks.clone()

HLTIterativeTrackingIter04 = cms.Sequence(hltIter4Merged)
HLTIterativeTrackingIter02 = cms.Sequence(hltIter2Merged)
HLTIterativeTracking = cms.Sequence(hltIter4Merged)
HLTIterativeTrackingForHighPt = cms.Sequence(hltIter4HighPtMerged)
HLTIterativeTrackingTau3Mu = cms.Sequence(hltIter4Tau3MuMerged)
HLTIterativeTrackingReg = cms.Sequence(hltIter4MergedReg)
HLTIterativeTrackingForElectronIter02 = cms.Sequence(hltIter2MergedForElectrons)
HLTIterativeTrackingForPhotonsIter02 = cms.Sequence(hltIter2MergedForPhotons)
HLTIterativeTrackingL3MuonIter02 = cms.Sequence(hltIter2L3MuonMerged)
HLTIterativeTrackingL3MuonRegIter02 = cms.Sequence(hltIter2L3MuonMergedReg)
HLTIterativeTrackingForBTagIter02 = cms.Sequence(hltIter2MergedForBTag)
HLTIterativeTrackingForTauIter02 = cms.Sequence(hltIter2MergedForTau)
HLTIterativeTrackingForTauIter04 = cms.Sequence(hltIter4MergedForTau)
HLTIterativeTrackingGlbTrkMuonIter02 = cms.Sequence(hltIter2GlbTrkMuonMerged)
HLTIterativeTrackingHighPtTkMu = cms.Sequence(hltIter2HighPtTkMuMerged)
HLTIterativeTrackingHighPtTkMuIsoIter02 = cms.Sequence(hltIter2HighPtTkMuIsoMerged)
HLTIterativeTrackingDisplacedJpsiIter02     = cms.Sequence(hltIter2DisplacedJpsiMerged)
HLTIterativeTrackingDisplacedPsiPrimeIter02 = cms.Sequence(hltIter2DisplacedPsiPrimeMerged)
HLTIterativeTrackingDisplacedNRMuMuIter02   = cms.Sequence(hltIter2DisplacedNRMuMuMerged)
HLTIterativeTrackingForBTagIter12 = cms.Sequence(hltIter2MergedForBTag)
HLTIterativeTrackingForBTagIteration0 = cms.Sequence(hltIter0PFlowTrackSelectionHighPurityForBTag)
HLTIterativeTrackingIteration4DisplacedJets = cms.Sequence(hltIter4MergedWithIter012DisplacedJets)
