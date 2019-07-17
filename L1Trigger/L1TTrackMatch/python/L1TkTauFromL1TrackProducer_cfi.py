import FWCore.ParameterSet.Config as cms


# --- to run L1EGCrystalClusterProducer, one needs the ECAL RecHits:
from Configuration.StandardSequences.Reconstruction_cff import *



L1TkTauFromL1Track = cms.EDProducer( 'L1TkTauFromL1Track' ,
                                  L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks")
)

