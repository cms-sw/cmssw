import FWCore.ParameterSet.Config as cms
process = cms.Process("OMTFTrainerProc")
import os
import sys
import commands
import re
from os import listdir
from os.path import isfile, join

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
    process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)
    process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False), 
                                         #SkipEvent = cms.untracked.vstring('ProductNotFound') 
                                     )

#inputGpFileNum = int(sys.argv[3])
'''
process.source = cms.Source(
    'PoolSource',
    #fileNames = cms.untracked.vstring('file:///afs/cern.ch/work/a/akalinow/CMS/OverlapTrackFinder/data/Crab/SingleMuFullEtaTestSample/720_FullEta_v1/data/SingleMu_16_p_1_2_TWz.root')
    #fileNames = cms.untracked.vstring('file:///home/akalinow/scratch/CMS/OverlapTrackFinder/Crab/SingleMuFullEta/721_FullEta_v4/data/SingleMu_25_p_133_2_QJ1.root')
    fileNames = cms.untracked.vstring('file:///afs/cern.ch/work/k/kbunkow/private/omtf_data/SingleMu_18_p_1_1_2KD.root')   
    #fileNames = cms.untracked.vstring('file:///afs/cern.ch/work/k/kbunkow/private/omtf_data/SingleMu_7_p_1_1_DkC.root')
    #fileNames = cms.untracked.vstring(set(chosenFiles))
    )
'''
path = '/eos/user/k/kbunkow/cms_data/SingleMuFullEta/721_FullEta_v4/'
#path = '/afs/cern.ch/work/k/kbunkow/public/data/SingleMuFullEta/721_FullEta_v4/'

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
#print onlyfiles

filesNameLike = sys.argv[2]
#chosenFiles = ['file://' + path + f for f in onlyfiles if (('_p_10_' in f) or ('_m_10_' in f))]
#chosenFiles = ['file://' + path + f for f in onlyfiles if (('_10_p_10_' in f))]
#chosenFiles = ['file://' + path + f for f in onlyfiles if (re.match('.*_._p_10.*', f))]
#chosenFiles = ['file://' + path + f for f in onlyfiles if ((filesNameLike in f))]

#print onlyfiles

chosenFiles = []

if filesNameLike == 'allPt' :
    for ptCode in range(31, 3, -1) :
        for sign in ['_m', '_p'] : #, m
            for i in range(1, 11, 1):
                for f in onlyfiles:
                   if (( '_' + str(ptCode) + sign + '_' + str(i) + '_') in f): 
                        #print f
                        chosenFiles.append('file://' + path + f) 
else :
    for i in range(20, 21, 1):
        for f in onlyfiles:
            if (( filesNameLike + '_' + str(i) + '_') in f): 
                print f
                chosenFiles.append('file://' + path + f) 
         

print "chosenFiles"
for chFile in chosenFiles:
    print chFile

firstEv = 0#40000
nEvents = 100000

# input files (up to 255 files accepted)
process.source = cms.Source('PoolSource',
fileNames = cms.untracked.vstring( 
    #'file:/eos/user/k/kbunkow/cms_data/SingleMuFullEta/721_FullEta_v4/SingleMu_16_p_1_1_xTE.root',
    #'file:/afs/cern.ch/user/k/kpijanow/Neutrino_Pt-2to20_gun_50.root',
    list(chosenFiles),
                                  ),
eventsToProcess = cms.untracked.VEventRange(
 '3:' + str(firstEv) + '-3:' +   str(firstEv + nEvents),
 '4:' + str(firstEv) + '-4:' +   str(firstEv + nEvents),
 '5:' + str(firstEv) + '-5:' +   str(firstEv + nEvents),
 '6:' + str(firstEv) + '-6:' +   str(firstEv + nEvents),
 '7:' + str(firstEv) + '-7:' +   str(firstEv + nEvents),
 '8:' + str(firstEv) + '-8:' +   str(firstEv + nEvents),
 '9:' + str(firstEv) + '-9:' +   str(firstEv + nEvents),
'10:' + str(firstEv) + '-10:' +  str(firstEv + nEvents),
'11:' + str(firstEv) + '-11:' +  str(firstEv + nEvents),
'12:' + str(firstEv) + '-12:' +  str(firstEv + nEvents),
'13:' + str(firstEv) + '-13:' +  str(firstEv + nEvents),
'14:' + str(firstEv) + '-14:' +  str(firstEv + nEvents),
'15:' + str(firstEv) + '-15:' +  str(firstEv + nEvents),
'16:' + str(firstEv) + '-16:' +  str(firstEv + nEvents),
'17:' + str(firstEv) + '-17:' +  str(firstEv + nEvents),
'18:' + str(firstEv) + '-18:' +  str(firstEv + nEvents),
'19:' + str(firstEv) + '-19:' +  str(firstEv + nEvents),
'20:' + str(firstEv) + '-20:' +  str(firstEv + nEvents),
'21:' + str(firstEv) + '-21:' +  str(firstEv + nEvents),
'22:' + str(firstEv) + '-22:' +  str(firstEv + nEvents),
'23:' + str(firstEv) + '-23:' +  str(firstEv + nEvents),
'24:' + str(firstEv) + '-24:' +  str(firstEv + nEvents),
'25:' + str(firstEv) + '-25:' +  str(firstEv + nEvents),
'26:' + str(firstEv) + '-26:' +  str(firstEv + nEvents),
'27:' + str(firstEv) + '-27:' +  str(firstEv + nEvents),
'28:' + str(firstEv) + '-28:' +  str(firstEv + nEvents),
'29:' + str(firstEv) + '-29:' +  str(firstEv + nEvents),
'30:' + str(firstEv) + '-30:' +  str(firstEv + nEvents),
'31:' + str(firstEv) + '-31:' +  str(firstEv + nEvents)),
skipEvents =  cms.untracked.uint32(0)
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10))

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
process.simOmtfDigis = cms.EDProducer("OMTFTrainer",
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
                                          patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0003.xml"),
                                          #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/expert/optimisedPats_thr_" + targetEffStr + ".xml"), #str(inputGpFileNum)
                                          optimisedPatsXmlFile = cms.string("Patterns_0x0003_TT.xml"), #"optimisedPats_" + str(gpNum) + ".xml"
                                          XMLDumpFileName = cms.string("TestEvents.xml"), 
                                          dumpResultToXML = cms.bool(False),
                                          dumpDetailedResultToXML = cms.bool(False),
                                          processorType = cms.string("OMTFProcessorTTMerger"),
                                          ttTracksSource = cms.string("SIM_TRACKS"),
                                          patternType = cms.string("GoldenPatternWithStat"), 
                                          refLayerMustBeValid = cms.bool(False),
                                          etaCutFrom = cms.double(0.82),
                                          etaCutTo = cms.double(1.24),
                                          #deltaPdf = cms.double(0.05),
                                          #ptRangeFrom = cms.double(401),
                                          #ptRangeTo   = cms.double(401),
                                          omtf = cms.PSet(
                                              configFromXML = cms.bool(False),   
                                              patternsXMLFiles = cms.VPSet(                                       
                                                cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0003.xml")), #must be here, otherwise some things in the OMTFConfiguration does not work should be FIXME
                                              ),
                                              #configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x00020005.xml"),
                                              configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x0006.xml"),
                                          ),
                                          
)

###Gen level filter configuration
#process.MuonEtaFilter = cms.EDFilter("SimTrackEtaFilter",
#                                 minNumber = cms.uint32(1),
#                                 src = cms.InputTag("g4SimHits"),
#                                 cut = cms.string("momentum.eta<1.24 && momentum.eta>0.83 &&  momentum.pt>1")
#                                )

#process.MuonEtaFilter*

#process.omtfAnalyzer= cms.EDAnalyzer("OmtfAnalyzer", 
#                                  outRootFile = cms.string("omtfAnalysis_" + str(inputGpFileNum) + ".root") )

process.L1TMuonSeq = cms.Sequence(process.esProd + process.simOmtfDigis ) #+ process.omtfAnalyzer 
 
process.L1TMuonPath = cms.Path(process.L1TMuonSeq)

process.schedule = cms.Schedule(process.L1TMuonPath)
