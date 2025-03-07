import FWCore.ParameterSet.Config as cms

# First register all the hit matching algorithms, then specify preferred ones at end.

# The stub windows used has been optimized for PU200 events
# We use by default the new modified tight tuning
# more details can be found in the following detector note: CMS DN-2020/005

# This script is adapted to the very last Tilted Tracker geometry (D76)
# This version was tested on CMSSW_13_3_0_pre2

   # PU200 new modified tight tuning
TTStubAlgorithm_official_Phase2TrackerDigi_ = cms.ESProducer("TTStubAlgorithm_official_Phase2TrackerDigi_",
       zMatchingPS  = cms.bool(True),
       zMatching2S  = cms.bool(True),
       NTiltedRings = cms.vdouble( 0., 12., 12., 12., 0., 0., 0.),
       BarrelCut    = cms.vdouble(0, 2.0, 2.5, 3.5, 4.0, 5.5, 6.5),
       TiltedBarrelCutSet = cms.VPSet(
           cms.PSet( TiltedCut = cms.vdouble( 0 ) ),
           cms.PSet( TiltedCut = cms.vdouble( 0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.5, 1.5, 1.0, 1.0) ),
           cms.PSet( TiltedCut = cms.vdouble( 0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.5, 2.5, 3.0, 3.0, 2.5, 2.5, 2.5) ),
           cms.PSet( TiltedCut = cms.vdouble(0, 4.0, 4.0, 4.0, 3.5, 3.5, 3.5, 3.0, 3.0, 2.5, 2.5, 2.5, 2.5) ),
       ),
       EndcapCutSet = cms.VPSet(
           cms.PSet( EndcapCut = cms.vdouble( 0 ) ),
           cms.PSet( EndcapCut = cms.vdouble(0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 4.0, 4.0, 2.5, 3.0, 3.5, 4.0, 5.0) ),
           cms.PSet( EndcapCut = cms.vdouble(0, 0.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 3.0, 3.5, 2.0, 2.5, 3.0, 4.0, 4.0) ),
           cms.PSet( EndcapCut = cms.vdouble(0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.5, 3.0, 3.5, 2.5, 2.5, 3.0, 3.5) ),
           cms.PSet( EndcapCut = cms.vdouble(0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 3.0) ),
           cms.PSet( EndcapCut = cms.vdouble(0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.5, 3.0, 2.0, 2.0, 2.5) ),
       )
)

# CBC3 hit matching algorithm
TTStubAlgorithm_cbc3_Phase2TrackerDigi_ = cms.ESProducer("TTStubAlgorithm_cbc3_Phase2TrackerDigi_",
   zMatchingPS = cms.bool(True),
   zMatching2S = cms.bool(True),
)


# Set the preferred hit matching algorithms.
# We prefer the global geometry algorithm for now in order not to break
# anything. Override with process.TTStubAlgorithm_PSimHit_ = ...,
# etc. in your configuration.
TTStubAlgorithm_Phase2TrackerDigi_ = cms.ESPrefer("TTStubAlgorithm_official_Phase2TrackerDigi_")