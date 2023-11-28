import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalGeometryTest")

process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load("Geometry.HcalCommonData.hcalDDDRecConstants_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi");
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("Geometry.EcalAlgo.ecalPreshowerCellParameterDump_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.EcalGeom=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('studyES.root')
                                   )

process.CaloGeometryBuilder.SelectedCalos = ['EcalBarrel', 'EcalEndcap', 'EcalPreshower']
process.ecalPreshowerCellParameterDump.debug = True

process.p1 = cms.Path(process.ecalPreshowerCellParameterDump)
