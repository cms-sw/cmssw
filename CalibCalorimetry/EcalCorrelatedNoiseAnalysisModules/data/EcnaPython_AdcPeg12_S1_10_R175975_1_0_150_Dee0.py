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
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/175/975/E023428B-43DB-E011-9085-BCAEC518FF89.root',
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/175/975/DE59CE6D-41DB-E011-8307-BCAEC5364C4C.root',
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/175/975/C8A54155-42DB-E011-8C22-BCAEC53296FB.root',
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/175/975/C05CE0A1-3EDB-E011-9A9C-0030486780B4.root',
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/175/975/BEF31C0B-95DB-E011-B882-003048D3756A.root',
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/175/975/4804F374-A1DB-E011-8442-BCAEC532971C.root',
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/175/975/1C15796C-3FDB-E011-A634-BCAEC518FF74.root',
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/175/975/0C9CA39C-40DB-E011-B7C0-BCAEC5329707.root',
        '/store/data/Run2011B/TestEnablesEcalHcalDT/RAW/v1/000/175/975/04491905-95DB-E011-B880-BCAEC518FF6E.root',    ),
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

