import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")


process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_P_V1::All"

process.load("CondCore.DBCommon.CondDBSetup_cfi")

#process.load("DQMServices.Core.DQM_cfg")


process.source = cms.Source("PoolSource",
#    debugFlag = cms.untracked.bool(True),
#    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(
      '/store/express/CRAFT09/StreamExpress/ALCARECO/v1/000/113/222/14019B7F-8899-DE11-8FCE-000423D9853C.root'
 
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
)


# if read from RAW
#process.load("EventFilter.DTRawToDigi.dtunpacker_cfi")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    authenticationMethod = cms.untracked.uint32(0),
    connect = cms.string('sqlite_file:noise.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTStatusFlagRcd'),
        tag = cms.string('noise')
    ))
)

process.noiseCalib = cms.EDAnalyzer("DTNoiseCalibration",
    #Define the wheel of interest (to set if fastAnalysis=false)
    wheel = cms.untracked.int32(0),
    #Define the sector of interest (to set if fastAnalysis=false)
    sector = cms.untracked.int32(11),
    #Database option (to set if cosmicRun=true)
    readDB = cms.untracked.bool(True),
    #The trigger width(TDC counts) (to set if cosmicRun=true and readDB=false)
    defaultTtrig = cms.untracked.int32(400),
    fastAnalysis = cms.untracked.bool(True),
    rootFileName = cms.untracked.string('DTNoiseCalib.root'),
    # Label to retrieve DT digis from the event
    # RAW: dtunpacker DIGI: muonDTDigis
    digiLabel = cms.untracked.string('muonDTDigis'), 
    #Trigger mode
    cosmicRun = cms.untracked.bool(True),
    debug = cms.untracked.bool(False),
    #The trigger width(ns) (to set if cosmicRun=false)
    TriggerWidth = cms.untracked.int32(25350),
    theOffset = cms.untracked.double(100.)
)

process.noiseComp = cms.EDAnalyzer("DTNoiseComputation",
    debug = cms.untracked.bool(False),
    fastAnalysis = cms.untracked.bool(False),
    #Total number of events	
    MaxEvents = cms.untracked.int32(200000),
    rootFileName = cms.untracked.string('DTNoiseCalib.root'),
    #Name of the ROOT file which will contains the
    newRootFileName = cms.untracked.string('DTNoiseComp.root')
)

#process.p = cms.Path(process.dtunpacker*process.noiseCalib)

process.p = cms.Path(process.noiseCalib)


