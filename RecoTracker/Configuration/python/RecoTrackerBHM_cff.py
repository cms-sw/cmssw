import FWCore.ParameterSet.Config as cms

#seeds
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForBeamHalo_cff import *
#Ckf
from RecoTracker.CkfPattern.CkfTrackCandidatesBHM_cff import *
#Final fit
from RecoTracker.TrackProducer.CTFFinalFitWithMaterialBHM_cff import *
tracksBeamHaloMuon = cms.Sequence(combinatorialbeamhaloseedfinder*ckfTrackCandidatesBeamHaloMuon*ctfWithMaterialTracksBeamHaloMuon)

