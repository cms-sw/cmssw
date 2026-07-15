"""
Read the MF geometry XML and dump to a root file for further test/inspection 
"""
import FWCore.ParameterSet.Config as cms

process = cms.Process("MFGEOMDUMP")
process.source = cms.Source("EmptySource")

# Create the MF geometry using DDD and as a detector geometry.
# this requires the ESSource be named XMLIdealGeometryESSource (otherwise TGeoMgrFromDdd fails)
# and the rootNodeName be different than cmsMagneticField:MAGF (otherwise XMLIdealGeometryESSource attaches the geometry to IdealMagneticFieldRecord)
from MagneticField.GeomBuilder.MFDDDGeometry_160812_cff import MFDDDGeometry
process.XMLIdealGeometryESSource = MFDDDGeometry
process.XMLIdealGeometryESSource.rootNodeName = 'cms:MCMS'

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.add_(cms.ESProducer("TGeoMgrFromDdd",
                            verbose = cms.untracked.bool(False),
                            level = cms.untracked.int32(14)
                            ))

process.dump = cms.EDAnalyzer("DumpSimGeometry", 
                              tag = cms.untracked.string("MFGeometry"),
                              outputFileName = cms.untracked.string("cmsDDMFGeometryTest.root"))

process.p = cms.Path(process.dump)
