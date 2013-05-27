import FWCore.ParameterSet.Config as cms

StackedTrackerGeometryESModule = cms.ESProducer("StackedTrackerGeometryESModule",
                                                truncation_precision = cms.uint32(2),
                                                z_window = cms.double(4.0),
                                                phi_window = cms.double(0.015),
                                                radial_window = cms.double(1.0),
                                                make_debug_file = cms.bool(True)
                                                )




