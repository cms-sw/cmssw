import FWCore.ParameterSet.Config as cms

process = cms.Process("CrystalCenterDump")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

# Geometry master configuration
#
process.load("SimG4CMS.EcalTestBeam.crystal248_cff")
process.load("Geometry.EcalTestBeam.TBH4GeometryXML_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.CaloGeometryBuilder.SelectedCalos = ['EcalBarrel']

process.myDump = cms.EDAnalyzer("CrystalCenterDump",
    Afac = cms.untracked.double(0.89),
    Bfac = cms.untracked.double(5.7),
    BeamEnergy = cms.untracked.double(120.)                                
)                              

process.p = cms.Path(process.myDump)
