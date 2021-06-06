#-------------- EcnaSystemPythoModuleInsert_1 / beginning
import FWCore.ParameterSet.Config as cms

process = cms.Process("ECNA")

#-------------- Message Logger
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        last_job_INFOS = cms.untracked.PSet(
            extension = cms.untracked.string('txt')
        )
    ),
    suppressInfo = cms.untracked.vstring('ecalEBunpacker')
)
#-------------- EcnaSystemPythoModuleInsert_1 / end
 
#-------------- Source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/178/231/D8EC3AC0-E1F3-E011-A784-003048678110.root',
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/178/231/A6B985E2-DEF3-E011-9A95-0030486780B8.root',
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/178/231/5CFCB4AF-DAF3-E011-ADCB-003048D3756A.root',
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/178/231/121DFE24-D7F3-E011-AD82-003048D2C020.root'
    ),
     duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
                            )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))
#-------------- EcnaSystemPythoModuleInsert_2_data / beginning
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")
process.ecalEBunpacker.InputLabel = cms.InputTag('hltEcalCalibrationRaw')

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
#-------------- EcnaSystemPythoModuleInsert_2 _data/ end

                                      #-------------- Getting Parameters
                                      sAnalysisName  = cms.string("AdcPeg12"),
                                      sNbOfSamples   = cms.string("10"),
                                      sFirstReqEvent = cms.string("1"),
                                      sLastReqEvent  = cms.string("0"),
                                      sReqNbOfEvts   = cms.string("150"),
                                      sStexName      = cms.string("Dee"),
                                      sStexNumber    = cms.string("0") 
                                      )
#-------------- EcnaSystemPythoModuleInsert_3 / beginning
process.p = cms.Path(process.ecalEBunpacker*process.myCnaPackage)
#-------------- EcnaSystemPythoModuleInsert_3 / end

