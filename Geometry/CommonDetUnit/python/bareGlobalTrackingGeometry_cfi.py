import FWCore.ParameterSet.Config as cms

#
# bareGlobalTrackingGeometry.cfi
# This cfi should be included to build the GlobalTrackingGeometry alone, 
# the user has to declare individual concrete geometries himself.
# (The file globalTrackingGeometry.cfi includes also all concrete geometries
# for muon and tracker)
#
GlobalTrackingGeometryESProducer = cms.ESProducer("GlobalTrackingGeometryESProducer")


