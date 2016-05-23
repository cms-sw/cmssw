import FWCore.ParameterSet.Config as cms
process = cms.Process("OMTFEmulation")
import os
import sys
import commands

verbose = False

process.load("FWCore.MessageLogger.MessageLogger_cfi")

if verbose: 
    process.MessageLogger = cms.Service("MessageLogger",
       suppressInfo       = cms.untracked.vstring('AfterSource', 'PostModule'),
       destinations   = cms.untracked.vstring(
                                             'detailedInfo'
                                             ,'critical'
                                             ,'cout'
                    ),
       categories = cms.untracked.vstring(
                                        'CondDBESSource'
                                        ,'EventSetupDependency'
                                        ,'Geometry'
                                        ,'MuonGeom'
                                        ,'GetManyWithoutRegistration'
                                        ,'GetByLabelWithoutRegistration'
                                        ,'Alignment'
                                        ,'SiStripBackPlaneCorrectionDepESProducer'
                                        ,'SiStripLorentzAngleDepESProducer'
                                        ,'SiStripQualityESProducer'
                                        ,'TRACKER'
                                        ,'HCAL'
        ),
       critical       = cms.untracked.PSet(
                        threshold = cms.untracked.string('ERROR') 
        ),
       detailedInfo   = cms.untracked.PSet(
                      threshold  = cms.untracked.string('INFO'), 
                      CondDBESSource  = cms.untracked.PSet (limit = cms.untracked.int32(0) ), 
                      EventSetupDependency  = cms.untracked.PSet (limit = cms.untracked.int32(0) ), 
                      Geometry  = cms.untracked.PSet (limit = cms.untracked.int32(0) ), 
                      MuonGeom  = cms.untracked.PSet (limit = cms.untracked.int32(0) ), 
                      Alignment  = cms.untracked.PSet (limit = cms.untracked.int32(0) ), 
                      GetManyWithoutRegistration  = cms.untracked.PSet (limit = cms.untracked.int32(0) ), 
                      GetByLabelWithoutRegistration  = cms.untracked.PSet (limit = cms.untracked.int32(0) ) 

       ),
       cout   = cms.untracked.PSet(
                threshold  = cms.untracked.string('INFO'), 
                CondDBESSource  = cms.untracked.PSet (limit = cms.untracked.int32(0) ), 
                EventSetupDependency  = cms.untracked.PSet (limit = cms.untracked.int32(0) ), 
                Geometry  = cms.untracked.PSet (limit = cms.untracked.int32(0) ), 
                MuonGeom  = cms.untracked.PSet (limit = cms.untracked.int32(0) ), 
                Alignment  = cms.untracked.PSet (limit = cms.untracked.int32(0) ), 
                GetManyWithoutRegistration  = cms.untracked.PSet (limit = cms.untracked.int32(0) ), 
                GetByLabelWithoutRegistration  = cms.untracked.PSet (limit = cms.untracked.int32(0) ) 
                ),
                                        )

if not verbose:
    process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(50000)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.source = cms.Source(
    'PoolSource',
    #fileNames = cms.untracked.vstring('file:///afs/cern.ch/work/a/akalinow/CMS/OverlapTrackFinder/data/Crab/SingleMuFullEtaTestSample/720_FullEta_v1/data/SingleMu_16_p_1_2_TWz.root')
    fileNames = cms.untracked.vstring('file:///home/akalinow/scratch/CMS/OverlapTrackFinder/Crab/SingleMuFullEta/721_FullEta_v4/data/SingleMu_25_p_133_2_QJ1.root')
   
)
'''
##Use all available events in a single job.
##Only for making the connections maps.
process.source.fileNames =  cms.untracked.vstring()
path = "/home/akalinow/scratch/CMS/OverlapTrackFinder/Crab/SingleMuFullEta/721_FullEta_v4/data/"
command = "ls "+path+"/SingleMu_10_?_*"
fileList = commands.getoutput(command).split("\n")
process.source.fileNames =  cms.untracked.vstring()
for aFile in fileList:
    process.source.fileNames.append('file:'+aFile)
'''

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000000))

###TEST
'''
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring('file:/home/akalinow/scratch/CMS/OverlapTrackFinder/Crab/SingleMuFullEtaTestSample/720_FullEta_v1/data/SingleMu_16_p_1_2_TWz.root'),
    #fileNames = cms.untracked.vstring('file:/home/akalinow/scratch/CMS/OverlapTrackFinder/Crab/JPsi_21kEvents.root')
    #eventsToProcess = cms.untracked.VEventRange('16:8')
    )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000))
'''
#######

###PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015_cff')
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
############################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

####Event Setup Producer
process.load('L1Trigger.L1TMuonOverlap.fakeOmtfParams_cff')
process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
      cms.PSet(record = cms.string('L1TMuonOverlapParamsRcd'),
               data = cms.vstring('L1TMuonOverlapParams'))
                   ),
   verbose = cms.untracked.bool(True)
)

###OMTF pattern maker configuration
process.omtfPatternMaker = cms.EDAnalyzer("OMTFPatternMaker",
                                          srcDTPh = cms.InputTag('simDtTriggerPrimitiveDigis'),
                                          srcDTTh = cms.InputTag('simDtTriggerPrimitiveDigis'),
                                          srcCSC = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
                                          srcRPC = cms.InputTag('simMuonRPCDigis'),                                              
                                          g4SimTrackSrc = cms.InputTag('g4SimHits'),
                                          makeGoldenPatterns = cms.bool(True),
                                          mergeXMLFiles = cms.bool(False),
                                          makeConnectionsMaps = cms.bool(False),                                      
                                          dropRPCPrimitives = cms.bool(False),                                    
                                          dropDTPrimitives = cms.bool(False),                                    
                                          dropCSCPrimitives = cms.bool(False),   
                                          ptCode = cms.int32(25),#this is old PAC pt scale.
                                          charge = cms.int32(1),
                                          omtf = cms.PSet(
                                              configFromXML = cms.bool(False),   
                                              patternsXMLFiles = cms.VPSet(                                       
                                                  cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x00020007.xml")),
                                              ),
                                              configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x00020005.xml"),
                                          )
)

###Gen level filter configuration
process.MuonEtaFilter = cms.EDFilter("SimTrackEtaFilter",
                                minNumber = cms.uint32(1),
                                src = cms.InputTag("g4SimHits"),
                                cut = cms.string("momentum.eta<1.24 && momentum.eta>0.83 &&  momentum.pt>1")
                                )

process.L1TMuonSeq = cms.Sequence(process.MuonEtaFilter*process.esProd*process.omtfPatternMaker)

process.L1TMuonPath = cms.Path(process.L1TMuonSeq)

process.schedule = cms.Schedule(process.L1TMuonPath)
