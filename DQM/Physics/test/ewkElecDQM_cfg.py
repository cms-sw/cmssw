import FWCore.ParameterSet.Config as cms



process = cms.Process("EwkDQM")
process.load("DQM.Physics.ewkElecDQM_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# load the full reconstraction configuration, to make sure we're getting all needed dependencies
process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.GeometryRecoDB_cff") #old one, to use for old releases
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = 'FT_53_V21_AN6::All'
#process.GlobalTag.globaltag = 'START70_V2::All'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
#    input = cms.untracked.int32(5000)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
##        '/store/relval/CMSSW_3_1_1/RelValWM/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/8E5D0675-E36B-DE11-8F71-001D09F242EF.root'

# MinBias real data!
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/196/3C9489A4-B5E8-DE11-A475-001D09F2A465.root',
    #'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/188/34641279-B5E8-DE11-A475-001D09F2910A.root',

# Real data
        #'/store/data/Run2012B/SingleElectron/AOD/22Jan2013-v1/30000/FE93DA20-837E-E211-8A41-002481E73676.root'
       # 'file:12251709-D77E-E211-96C8-003048F118FE.root' # data
       #   , 'file:5072427B-407E-E211-88EF-003048F237FE.root' #data
        #  'file:DEC5AD62-280C-E311-89A7-002618FDA216.root'
    # 'file:/tmp/andriusj/ZeePU.root'
     'file:/tmp/andriusj/Data2012D_DoubleEl.root'
     )
)

runOnData = False

#process.dqmEnv.subSystemFolder = 'SMP'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = cms.untracked.string('/Physics/EWK/Elec')
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True) 
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

if runOnData:
     process.dqmSaver.saveByRun = cms.untracked.int32(1)
     process.dqmSaver.saveAtJobEnd =cms.untracked.bool(False) 
     process.dqmSaver.forceRunNumber = cms.untracked.int32(-1)


process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo'),
    detailedInfo = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(100) ),
            threshold = cms.untracked.string('DEBUG')
            #threshold = cms.untracked.string('INFO')
            #threshold = cms.untracked.string('ERROR')
    )
)
#process.ana = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.ewkElecDQM+process.dqmSaver)

