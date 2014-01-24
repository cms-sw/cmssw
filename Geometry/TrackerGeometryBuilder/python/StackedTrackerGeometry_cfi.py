import FWCore.ParameterSet.Config as cms

StackedTrackerGeometryESModule = cms.ESProducer( "StackedTrackerGeometryESModule",
                                                 truncation_precision = cms.uint32(2),
                                                 z_window = cms.double(4.0),
                                                 phi_window = cms.double(0.015),
                                                 radial_window = cms.double(1.0),
                                                 make_debug_file = cms.bool(True),

# Extras for CBC3 chip
#                                                 partitionsPerRoc = cms.int32(4),
                                                 CBC3_MaxStubs = cms.uint32(3),
# Double tab2013 table as CBC3 chip uses full width -- this table for 7-ring design
#   BarrelCut = cms.vdouble( 0, 5, 5, 6, 9, 11, 14 ), #Use 0 as dummy to have direct access using DetId to the correct element
#   EndcapCutSet = cms.VPSet( #Double the tab2013 table as CBC3 chip uses full width
#     cms.PSet( EndcapCut = cms.vdouble( 0 ) ), #Use 0 as dummy to have direct access using DetId to the correct element
#     cms.PSet( EndcapCut = cms.vdouble( 0, 3, 3, 4, 4, 5, 6, 6, 7, 5, 6, 6, 9, 11, 11 ) ), #D1
#     cms.PSet( EndcapCut = cms.vdouble( 0, 2, 3, 4, 4, 5, 5, 6, 7, 4, 6, 6, 8, 8, 10 ) ), #D2 ...
#     cms.PSet( EndcapCut = cms.vdouble( 0, 2, 2, 3, 4, 4, 5, 6, 6, 4, 4, 6, 6, 8, 9 ) ),
#     cms.PSet( EndcapCut = cms.vdouble( 0, 2, 2, 3, 3, 4, 4, 5, 5, 7, 4, 5, 6, 6, 8 ) ),
#     cms.PSet( EndcapCut = cms.vdouble( 0, 2, 3, 3, 4, 4, 5, 5, 6, 8, 4, 5, 6, 7 ) ),
#     cms.PSet( EndcapCut = cms.vdouble( 0, 3, 3, 3, 4, 4, 5, 6, 7, 4, 5, 6, 6 ) ),
#     cms.PSet( EndcapCut = cms.vdouble( 0, 2, 3, 4, 4, 5, 5, 6, 8, 4, 5, 6 ) ) ),
                                                # missing rings are not taken into account in numbering, so everything
                                                # always starts from 1 to N, with increasing r

# Double tab2013 table as CBC3 chip uses full width -- this table for 5-ring design up to 6_2_0_SLHC5
#   BarrelCut = cms.vdouble( 0, 5, 5, 6, 9, 11, 13 ), #Use 0 as dummy to have direct access using DetId to the correct element
#   EndcapCutSet = cms.VPSet(
#     cms.PSet( EndcapCut = cms.vdouble( 0 ) ), #Use 0 as dummy to have direct access using DetId to the correct element
#     cms.PSet( EndcapCut = cms.vdouble( 0, 3, 3, 4, 4, 4, 5, 6, 6, 7, 5, 6, 7, 8, 10, 11 ) ), #D1
#     cms.PSet( EndcapCut = cms.vdouble( 0, 3, 3, 3, 4, 4, 5, 5, 6, 6, 4, 5, 6, 7, 8, 9 ) ), #D2 ...
#     cms.PSet( EndcapCut = cms.vdouble( 0, 3, 3, 4, 4, 4, 4, 5, 5, 4, 4, 6, 6, 7, 8, 9 ) ),
#     cms.PSet( EndcapCut = cms.vdouble( 0, 3, 4, 4, 4, 4, 5, 6, 4, 4, 5, 6, 7, 7, 8, 9 ) ),
#     cms.PSet( EndcapCut = cms.vdouble( 0, 3, 4, 4, 4, 4, 6, 6, 4, 4, 5, 6, 7, 7, 8, 9 ) ) ),
                                                # missing rings are not taken into account in numbering, so everything
                                                # always starts from 1 to N, with increasing r

# Double tab2013 table as CBC3 chip uses full width -- this table for 5-ring design since 6_2_0_SLHC5
   BarrelCut = cms.vdouble( 0, 5, 5, 6, 9, 11, 13 ), #Use 0 as dummy to have direct access using DetId to the correct element
   EndcapCutSet = cms.VPSet(
     cms.PSet( EndcapCut = cms.vdouble( 0 ) ), #Use 0 as dummy to have direct access using DetId to the correct element
     cms.PSet( EndcapCut = cms.vdouble( 0, 4, 4, 4, 4, 5, 5, 5, 6, 7, 9, 6, 7, 8, 9, 10 ) ), #D1
     cms.PSet( EndcapCut = cms.vdouble( 0, 3, 4, 4, 4, 4, 5, 5, 5, 6, 8, 5, 6, 7, 8, 9 ) ), #D2 ...
     cms.PSet( EndcapCut = cms.vdouble( 0, 3, 4, 4, 4, 4, 4, 5, 5, 5, 7, 8, 5, 6, 7, 8 ) ),
     cms.PSet( EndcapCut = cms.vdouble( 0, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7, 5, 5, 6, 7 ) ),
     cms.PSet( EndcapCut = cms.vdouble( 0, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 5, 5, 6 ) ) ),

)

