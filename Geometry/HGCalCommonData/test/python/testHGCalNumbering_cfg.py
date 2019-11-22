import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
#process.load("Geometry.HGCalCommonData.testHGCV8XML_cfi")
#process.load("Geometry.CMSCommonData.cmsExtendedGeometry2026D35XML_cfi")
#process.load("Geometry.CMSCommonData.cmsExtendedGeometry2026D41XML_cfi")
#process.load("Geometry.CMSCommonData.cmsExtendedGeometry2026D46XML_cfi")
process.load("Geometry.HGCalCommonData.testHGCXML_cfi")
process.load("Geometry.HGCalCommonData.hgcalParametersInitialization_cfi")
process.load("Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi")
#process.load("Geometry.HGCalCommonData.hgcalV6ParametersInitialization_cfi")
#process.load("Geometry.HGCalCommonData.hgcalV6NumberingInitialization_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HGCalGeom')

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
                                   PGunParameters = cms.PSet(
                                       PartID = cms.vint32(14),
                                       MinEta = cms.double(-3.5),
                                       MaxEta = cms.double(3.5),
                                       MinPhi = cms.double(-3.14159265359),
                                       MaxPhi = cms.double(3.14159265359),
                                       MinE   = cms.double(9.99),
                                       MaxE   = cms.double(10.01)
                                   ),
                                   AddAntiParticle = cms.bool(False),
                                   Verbosity       = cms.untracked.int32(0),
                                   firstRun        = cms.untracked.uint32(1)
                               )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.prodEE = cms.EDAnalyzer("HGCalNumberingTester",
                                NameSense     = cms.string("HGCalEESensitive"),
                                NameDevice    = cms.string("HGCal EE"),
                                LocalPositionX= cms.vdouble(500.0,350.0,800.0,1400.0),
                                LocalPositionY= cms.vdouble(500.0,0.0,0.0,0.0),
                                Increment     = cms.int32(19),
#                               DetType       = cms.int32(1),
                                DetType       = cms.int32(2),
                                Reco          = cms.bool(False)
)

process.prodHEF = process.prodEE.clone(
    NameSense  = "HGCalHESiliconSensitive",
    NameDevice = "HGCal HE Front",
    Increment  = 9
)
 
process.prodHEB = process.prodEE.clone(
    NameSense  = "HGCalHEScintillatorSensitive",
    NameDevice = "HGCal HE Back",
    Increment  = 9,
    LocalPositionX= [1000.0,1400.0,1500.0,1600.0],
    LocalPositionY= [1000.0,0.0,0.0,0.0],
    DetType    = 0
)
 
#process.p1 = cms.Path(process.generator*process.prodEE*process.prodHEF)
process.p1 = cms.Path(process.generator*process.prodEE*process.prodHEF*process.prodHEB)
