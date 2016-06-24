import FWCore.ParameterSet.Config as cms

# First register all the hit matching algorithms, then specify preferred ones at end.

# Tab2013 hit matching algorithm
TTStubAlgorithm_official_Phase2TrackerDigi_ = cms.ESProducer("TTStubAlgorithm_official_Phase2TrackerDigi_",
   zMatchingPS = cms.bool(False),
   zMatching2S = cms.bool(True),
   # 2GeV/c Threshold
   # --> Keep 99% of the stubs between 2 and 10 GeV/c
   BarrelCut = cms.vdouble( 0, 1.5, 1.5, 2.5, 4, 5, 6.5), 
   EndcapCutSet = cms.VPSet(
        cms.PSet( EndcapCut = cms.vdouble( 0 ) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 4, 2.5, 3, 3.5, 4, 5) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 1, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3.5, 2, 2.5, 3, 3.5, 4) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 2, 3, 3.5, 2, 2.5, 3, 3.5) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 1, 1, 1, 1, 1, 1.5, 2, 2, 2, 2.5, 3, 2, 2, 2.5, 3) ),
        cms.PSet( EndcapCut = cms.vdouble( 0, 1, 1, 1, 1, 1, 1.5, 1.5, 2, 2, 2, 3, 3, 2, 2, 2.5) ),
        )
)

# CBC3 hit matching algorithm
TTStubAlgorithm_cbc3_Phase2TrackerDigi_ = cms.ESProducer("TTStubAlgorithm_cbc3_Phase2TrackerDigi_",
   zMatchingPS = cms.bool(False),
   zMatching2S = cms.bool(True),
)


# Set the preferred hit matching algorithms.
# We prefer the global geometry algorithm for now in order not to break
# anything. Override with process.TTStubAlgorithm_PSimHit_ = ...,
# etc. in your configuration.
TTStubAlgorithm_Phase2TrackerDigi_ = cms.ESPrefer("TTStubAlgorithm_official_Phase2TrackerDigi_")

