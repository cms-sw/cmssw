import FWCore.ParameterSet.Config as cms

print 'Calling slhc geometry'
# Ideal geometry, needed for simulation
from SLHCUpgradeSimulations.Geometry.Phase1_R30F12_cmsSimIdealGeometryXML_cff import *
from Geometry.TrackerNumberingBuilder.trackerNumberingSLHCGeometry_cfi import *
