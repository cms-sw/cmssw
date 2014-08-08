# produce pixel cluster & rechits from digia
# works directly or through raw
# 
#
##############################################################################

import FWCore.ParameterSet.Config as cms

process = cms.Process("RecHitTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Services_cff")

# clusterizer 
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")

# for raw
#process.load("EventFilter.SiPixelRawToDigi.SiPixelDigiToRaw_cfi")
#process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')


# needed for pixel RecHits (templates?)
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('SiPixelClusterizer'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
#       threshold = cms.untracked.string('INFO')
        threshold = cms.untracked.string('WARNING')
#       threshold = cms.untracked.string('ERROR')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)
# get the files from DBS:
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
#    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100/clus/clus1.root'
    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/rechits/rechits2_mc71.root'
#    '/store/relval/CMSSW_7_1_0_pre7/RelValProdMinBias/GEN-SIM-RECO/PRE_STA71_V3-v1/00000/9E55469D-B2D1-E311-BEA8-02163E00B4B7.root'

#"/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/F60495B3-1E41-E211-BB7C-003048D3756A.root",

##"/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/F2BA6B22-2C41-E211-9D7A-003048D2BED6.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/E4E6B318-2041-E211-B351-001D09F29114.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/B27AC385-3241-E211-AD10-0019B9F4A1D7.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/AC6EF0B7-4941-E211-9EFB-003048D374F2.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/AA4018D3-2C41-E211-8279-00215AEDFD98.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/A8CF653C-4D41-E211-811E-003048673374.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/98EEEB5E-4A41-E211-A591-001D09F25460.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/96BE2949-2241-E211-9993-001D09F23F2A.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/90F6F479-2641-E211-99E5-001D09F29524.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/8AAFC294-2141-E211-89E8-003048F1182E.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/82B885ED-2241-E211-9877-001D09F252E9.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/6440884D-2941-E211-BBA9-0025901D6288.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/604E8D2C-2741-E211-B542-003048F11C28.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/4EB6D745-2241-E211-9738-001D09F24D8A.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/3E863C44-2241-E211-9255-001D09F25041.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/3E76F7E8-2741-E211-8249-003048D37666.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/3C1D83B8-3641-E211-8C66-0025B32035A2.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/2A12A045-2241-E211-8BF5-001D09F2915A.root",
## "/store/data/Run2012D/MinimumBias/RECO/PromptReco-v1/000/208/686/1661335F-3041-E211-9B96-00237DDBE0E2.root",

  )
)

#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('208686:73-208686:463')

# a service to use root histos
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root')
)

# Choose the global tag here:
#process.GlobalTag.globaltag = "MC_70_V1::All"
#process.GlobalTag.globaltag = "START70_V1::All"
#process.GlobalTag.globaltag = "POSTLS170_V5::All"
process.GlobalTag.globaltag = "START71_V1::All"
#process.GlobalTag.globaltag = "MC_71_V1::All"
#process.GlobalTag.globaltag = "POSTLS171_V1::All"
#process.GlobalTag.globaltag = "PRE_MC_71_V2::All"
# for data
#process.GlobalTag.globaltag = "GR_R_71_V1::All"

# DB stuff 
# GenError
useLocalDB = True
if useLocalDB :
    process.DBReaderFrontier = cms.ESSource("PoolDBESSource",
     DBParameters = cms.PSet(
         messageLevel = cms.untracked.int32(0),
         authenticationPath = cms.untracked.string('')
     ),
     toGet = cms.VPSet(
 	 cms.PSet(
          record = cms.string('SiPixelGenErrorDBObjectRcd'),
#          tag = cms.string('SiPixelGenErrorDBObject38TV10')
          tag = cms.string('SiPixelGenErrorDBObject_38T_v1_mc')
 	 ),
 	),
#     connect = cms.string('sqlite_file:siPixelGenErrors38T.db')
#     connect = cms.string('frontier://FrontierProd/CMS_COND_PIXEL_000')
     connect = cms.string('frontier://FrontierPrep/CMS_COND_PIXEL')
    ) # end process
# endif

useLocalDB2 = False
if useLocalDB2 :
    process.DBReaderFrontier2 = cms.ESSource("PoolDBESSource",
     DBParameters = cms.PSet(
         messageLevel = cms.untracked.int32(0),
         authenticationPath = cms.untracked.string('')
     ),
     toGet = cms.VPSet(
 		cms.PSet(
 			record = cms.string("SiPixelTemplateDBObjectRcd"),
 			tag = cms.string("SiPixelTemplateDBObject38TV10")
# 			tag = cms.string("SiPixelTemplateDBObject38Tv21")
 		),
 	),
     connect = cms.string('sqlite_file:siPixelTemplates38T.db')
#     connect = cms.string('frontier://FrontierPrep/CMS_COND_PIXEL')
#     connect = cms.string('frontier://FrontierProd/CMS_COND_31X_PIXEL')
   ) # end process
# endif
 
process.myprefer = cms.ESPrefer("PoolDBESSource","DBReaderFrontier")
#process.myprefer2 = cms.ESPrefer("PoolDBESSource","DBReaderFrontier2")


#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#    DBParameters = cms.PSet(
#        messageLevel = cms.untracked.int32(0),
#        authenticationPath = cms.untracked.string('')
#    ),
#    timetype = cms.string('runnumber'),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('SiPixelQualityRcd'),
#        tag = cms.string('SiPixelBadModule_test')
#    )),
#    connect = cms.string('sqlite_file:test.db')
#)
#
# To use a test DB instead of the official pixel object DB tag: 
#process.customDead = cms.ESSource("PoolDBESSource", process.CondDBSetup, connect = cms.string('sqlite_file:/afs/cern.ch/user/v/vesna/Digitizer/dead_20100901.db'), toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelQualityRcd'), tag = cms.string('dead_20100901'))))
#process.es_prefer_customDead = cms.ESPrefer("PoolDBESSource","customDead")


process.o1 = cms.OutputModule("PoolOutputModule",
                              outputCommands = cms.untracked.vstring('drop *','keep *_*_*_RecHitTest'),
            fileName = cms.untracked.string('file:clus.root')
#            fileName = cms.untracked.string('file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/rechits/rechits2_mc71.root')
)

#process.Timing = cms.Service("Timing")
#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

# My 
# modify clusterie parameters
#process.siPixelClusters.ClusterThreshold = 4000.0

# DIRECT
# direct clusterization (no raw step)
# label of digis 
process.siPixelClusters.src = 'simSiPixelDigis'

# read rechits
process.analysis = cms.EDAnalyzer("ReadPixelRecHit",
    Verbosity = cms.untracked.bool(False),
    src = cms.InputTag("siPixelRecHits"),
)

# plus pixel clusters  (OK)
#process.p1 = cms.Path(process.siPixelClusters)
# plus pixel rechits (OK)
#process.p1 = cms.Path(process.pixeltrackerlocalreco*process.analysis)
# only rechits
process.p1 = cms.Path(process.siPixelRecHits*process.analysis)

# RAW
# clusterize through raw (OK)
# for Raw2digi for simulations 
#process.siPixelRawData.InputLabel = 'mix'
#process.siPixelDigis.InputLabel = 'siPixelRawData'
# process.siStripDigis.ProductLabel = 'SiStripDigiToRaw'
#process.siPixelClusters.src = 'siPixelDigis'

#process.p1 = cms.Path(process.siPixelRawData)
#process.p1 = cms.Path(process.siPixelRawData*process.siPixelDigis)
#process.p1 = cms.Path(process.siPixelRawData*process.siPixelDigis*process.pixeltrackerlocalreco)

# save output 
#process.outpath = cms.EndPath(process.o1)
