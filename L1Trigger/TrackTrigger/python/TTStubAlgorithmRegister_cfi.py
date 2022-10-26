import FWCore.ParameterSet.Config as cms

# First register all the hit matching algorithms, then specify preferred ones at end.

# The stub windows used has been optimized for for PU200 events
# We use by default the tight tuning
#
# Definition is presented here:
#
# https://indico.cern.ch/event/681577/#4-update-of-the-track-trigger
#
# This script is adapted to the very last Tilted Tracker geometry to date (tracker T5)
# This version was tested on CMSSW 10_0_0_pre1
# 

TTStubAlgorithm_official_Phase2TrackerDigi_ = cms.ESProducer("TTStubAlgorithm_official_Phase2TrackerDigi_",
   zMatchingPS  = cms.bool(True),
   zMatching2S  = cms.bool(True),
   #Number of tilted rings per side in barrel layers (for tilted geom only)
   NTiltedRings = cms.vdouble( 0., 12., 12., 12., 0., 0., 0.), 
   # PU200 tight tuning, optimized for muons
   BarrelCut    = cms.vdouble( 0, 2, 2.5, 3.5, 4.5, 5.5, 7),
   TiltedBarrelCutSet = cms.VPSet(
        cms.PSet( TiltedCut = cms.vdouble( 0 ) ),
        cms.PSet( TiltedCut = cms.vdouble( 0, 3, 3, 2.5, 3, 3, 2.5, 2.5, 2, 1.5, 1.5, 1, 1) ),
        cms.PSet( TiltedCut = cms.vdouble( 0, 3.5, 3, 3, 3, 3, 2.5, 2.5, 3, 3, 2.5, 2.5, 2.5) ),
        cms.PSet( TiltedCut = cms.vdouble( 0, 4, 4, 4, 3.5, 3.5, 3.5, 3.5, 3, 3, 3, 3, 3) ),
   	),
   EndcapCutSet = cms.VPSet(
        cms.PSet( EndcapCut = cms.vdouble( 0 ) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 1, 2.5, 2.5, 3, 2.5, 3, 3.5, 4, 4, 4.5, 3.5, 4, 4.5, 5, 5.5) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 0.5, 2.5, 2.5, 3, 2.5, 3, 3, 3.5, 3.5, 4, 3.5, 3.5, 4, 4.5, 5) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 1, 3, 3, 2.5, 3.5, 3.5, 3.5, 4, 3.5, 3.5, 4, 4.5) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 1, 2.5, 3, 2.5, 3.5, 3, 3, 3.5, 3.5, 3.5, 4, 4) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 0.5, 1.5, 3, 2.5, 3.5, 3, 3, 3.5, 4, 3.5, 4, 3.5) ),
        )

   # PU200 loose tuning, optimized for robustness (uncomment if you want to use it)
   #BarrelCut    = cms.vdouble( 0, 2.0, 3, 4.5, 6, 6.5, 7.0),
   #TiltedBarrelCutSet = cms.VPSet(
   #     cms.PSet( TiltedCut = cms.vdouble( 0 ) ),
   #     cms.PSet( TiltedCut = cms.vdouble( 0, 3, 3., 2.5, 3., 3., 2.5, 2.5, 2., 1.5, 1.5, 1, 1) ),
   #     cms.PSet( TiltedCut = cms.vdouble( 0, 4., 4, 4, 4, 4., 4., 4.5, 5, 4., 3.5, 3.5, 3) ),
   #     cms.PSet( TiltedCut = cms.vdouble( 0, 5, 5, 5, 5, 5, 5, 5.5, 5, 5, 5.5, 5.5, 5.5) ),
   #	),
   #EndcapCutSet = cms.VPSet(
   #     cms.PSet( EndcapCut = cms.vdouble( 0 ) ),
   #     cms.PSet( EndcapCut = cms.vdouble( 0, 1., 2.5, 2.5, 3.5, 5.5, 5.5, 6, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 7, 7) ),
   #     cms.PSet( EndcapCut = cms.vdouble( 0, 0.5, 2.5, 2.5, 3, 5, 6, 6, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 7, 7) ),
   #     cms.PSet( EndcapCut = cms.vdouble( 0, 1, 3., 4.5, 6., 6.5, 6.5, 6.5, 7, 7, 7, 7, 7) ),
   #     cms.PSet( EndcapCut = cms.vdouble( 0, 1., 2.5, 3.5, 6., 6.5, 6.5, 6.5, 6.5, 7, 7, 7, 7) ),
   #     cms.PSet( EndcapCut = cms.vdouble( 0, 0.5, 1.5, 3., 4.5, 6.5, 6.5, 7, 7, 7, 7, 7, 7) ),
   #     )
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

