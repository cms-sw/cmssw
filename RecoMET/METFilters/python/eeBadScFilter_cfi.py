
import FWCore.ParameterSet.Config as cms

eeBadScFilter = cms.EDFilter(
  "EEBadScFilter",
  # the EE rechit collection found in AOD
  EERecHitSource = cms.InputTag('reducedEcalRecHitsEE'),
  # minimum rechit energy used for rechit flag check 
  EminHit        = cms.double(1000.),
  # minimum transverse energy of the 5x5 array for each bad SC 
  EtminSC        = cms.double(1000.),
  # size of the crystal array (default = 5 -> 5x5 array)
  SCsize         = cms.int32(5),
  # minimum number of hits above EminHit with !kGood flags
  nBadHitsSC     = cms.int32(2),
  #coordinates of the crystals in the centre of each bad supercrystal
  # packed into a single integer in the form  iz,ix,iy
  #   for instance -1023023 ->  ix=23, iy=23, iz=-1
  badscEE        = cms.vint32(-1023023,1048098,-1078063,-1043093),
  taggingMode = cms.bool(False),
  #prints debug info for each supercrystal if set to true
  debug = cms.bool(False),
)
