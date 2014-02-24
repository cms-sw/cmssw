import FWCore.ParameterSet.Config as cms

# First register all the hit matching algorithms, then specify preferred ones at end.

# Hit matching algorithm a
TTStubAlgorithm_a_PixelDigi_ = cms.ESProducer("TTStubAlgorithm_a_PixelDigi_")

# Global geometry hit matching algorithm
TTStubAlgorithm_globalgeometry_PixelDigi_ = cms.ESProducer("TTStubAlgorithm_globalgeometry_PixelDigi_",
    ipWidth = cms.double(200.0),
    minPtThreshold = cms.double(2.0),
)

# Pixel ray hit matching algorithm
TTStubAlgorithm_pixelray_PixelDigi_ = cms.ESProducer("TTStubAlgorithm_pixelray_PixelDigi_",
    ipWidth = cms.double(40.0),
    minPtThreshold = cms.double(2.0),
)

# Window hit matching algorithm
TTStubAlgorithm_window_PixelDigi_ = cms.ESProducer("TTStubAlgorithm_window_PixelDigi_",
    ipWidth = cms.double(15.0),
    minPtThreshold = cms.double(2.0),
    RowResolution = cms.double(1.0),
    ColResolution = cms.double(1.0)
)

# Window2012 hit matching algorithm
TTStubAlgorithm_window2012_PixelDigi_ = cms.ESProducer("TTStubAlgorithm_window2012_PixelDigi_",
    minPtThreshold = cms.double(2.0),
)

# Window2013 hit matching algorithm
TTStubAlgorithm_window2013_PixelDigi_ = cms.ESProducer("TTStubAlgorithm_window2013_PixelDigi_",
   minPtThreshold = cms.double(2.0),
   zMatchingPS = cms.bool(False),
   zMatching2S = cms.bool(True),
)

# Tab2013 hit matching algorithm
TTStubAlgorithm_tab2013_PixelDigi_ = cms.ESProducer("TTStubAlgorithm_tab2013_PixelDigi_",
   zMatchingPS = cms.bool(False),
   zMatching2S = cms.bool(True),
   BarrelCut = cms.vdouble( 0, 2.5, 2.5, 3.0, 4.5, 5.5, 6.5 ), #Use 0 as dummy to have direct access using DetId to the correct element
   EndcapCutSet = cms.VPSet(
     cms.PSet( EndcapCut = cms.vdouble( 0 ) ), #Use 0 as dummy to have direct access using DetId to the correct element
     cms.PSet( EndcapCut = cms.vdouble( 0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 3.0, 3.5, 4.5, 3.0, 3.5, 4.0, 4.5, 5.0 ) ), #D1
     cms.PSet( EndcapCut = cms.vdouble( 0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 3.0, 4.0, 2.5, 3.0, 3.5, 4.0, 4.5 ) ), #D2 ...
     cms.PSet( EndcapCut = cms.vdouble( 0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 3.5, 4.0, 2.5, 3.0, 3.5, 4.0 ) ),
     cms.PSet( EndcapCut = cms.vdouble( 0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 3.0, 3.5, 2.5, 2.5, 3.0, 3.5 ) ),
     cms.PSet( EndcapCut = cms.vdouble( 0, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 3.0, 3.5, 2.5, 2.5, 3.0 ) ), # missing rings are not taken into account in numbering, so everything
                                                                                                      # always starts from 1 to N, with increasing r
   )
)

# CBC3 hit matching algorithm
TTStubAlgorithm_cbc3_PixelDigi_ = cms.ESProducer("TTStubAlgorithm_cbc3_PixelDigi_",
   zMatchingPS = cms.bool(False),
   zMatching2S = cms.bool(True),
)


# Set the preferred hit matching algorithms.
# We prefer the global geometry algorithm for now in order not to break
# anything. Override with process.TTStubAlgorithm_PSimHit_ = ...,
# etc. in your configuration.
TTStubAlgorithm_PixelDigi_ = cms.ESPrefer("TTStubAlgorithm_window2013_PixelDigi_")

