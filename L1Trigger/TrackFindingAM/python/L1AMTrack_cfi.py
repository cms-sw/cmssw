import FWCore.ParameterSet.Config as cms


TTTracksFromPixelDigisAM = cms.EDProducer("TrackFindingAMProducer",
   TTStubsBricks = cms.InputTag("TTStubsFromPixelDigis", "StubAccepted"),
   NumSectors = cms.int32(6),
   NumWedges = cms.int32(3),
   inputBankFile = cms.string('/afs/cern.ch/work/s/sviret/testarea/PatternBanks/BE_5D/Eta7_Phi8/ss32_cov40/612_SLHC6_MUBANK_lowmidhig_sec37_ss32_cov40.pbk'),
   threshold     = cms.int32(5)
)

#process.BeamSpotFromSim = cms.EDProducer("BeamSpotFromSimProducer")
#process.TrackFindingTracklet_step = cms.Path(process.BeamSpotFromSim*process.TTTracksFromPixelDigisTracklet)

