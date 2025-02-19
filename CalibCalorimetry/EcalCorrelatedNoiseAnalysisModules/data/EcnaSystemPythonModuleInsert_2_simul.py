#-------------- EcnaSystemPythoModuleInsert_2_simul / beginning
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")
process.ecalEBunpacker.InputLabel = cms.string('ecalPacker')

# ECAL Geometry:
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.EcalCommonData.EcalOnly_cfi")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

#-------------- module for the CNA test
process.myCnaPackage = cms.EDAnalyzer("EcnaAnalyzer",
                                      digiProducer = cms.string("ecalEBunpacker"),
                                      #-------------- Getting Event Header
                                      eventHeaderProducer = cms.string("ecalEBunpacker"),
                                      eventHeaderCollection = cms.string(""),
                                      #-------------- Getting Digis
                                      EBdigiCollection = cms.string("ebDigis"),
                                      EEdigiCollection = cms.string("eeDigis"),
#-------------- EcnaSystemPythoModuleInsert_2 _simul/ end
