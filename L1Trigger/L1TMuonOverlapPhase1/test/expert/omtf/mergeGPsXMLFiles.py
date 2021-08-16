import FWCore.ParameterSet.Config as cms
process = cms.Process("OMTFEmulation")
import os
import sys
import commands

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring('file:/home/akalinow/scratch/CMS/OverlapTrackFinder/Crab/SingleMuFullEtaTestSample/720_FullEta_v1/data/SingleMu_16_p_1_2_TWz.root')
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

###PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015_cff')
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
############################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

path = "/home/akalinow/scratch/CMS/OverlapTrackFinder/Emulator/job_3_pat/8_0_9_20_06_2016/"

# Strip XML files from heading and trailing OMTF tags
sedCommand = "sed '/OMTF/d' "
command = "echo \<OMTF version=\\\"0x0004\\\"\> > mergedPatterns.xml"
os.system(command)
for ipt in xrange(4,32):

    fileName = path+"SingleMu_"+str(ipt)+"_p/GPs.xml"
    outputFileName = path+"SingleMu_"+str(ipt)+"_p/GPs_stripped.xml"
    command = sedCommand+fileName+" > "+outputFileName
    os.system(command)

    command = "cat "+outputFileName+" >> mergedPatterns.xml"
    os.system(command)
    
    fileName = path+"SingleMu_"+str(ipt)+"_m/GPs.xml"
    outputFileName = path+"SingleMu_"+str(ipt)+"_m/GPs_stripped.xml"
    command = sedCommand+fileName+" > "+outputFileName
    os.system(command)

    command = "cat "+outputFileName+" >> mergedPatterns.xml"
    os.system(command)

command = "echo \<\\/OMTF\> >> mergedPatterns.xml"
os.system(command)

process.load('L1Trigger.L1TMuonOverlap.fakeOmtfParams_cff')

process.omtfParams.patternsXMLFiles = cms.VPSet(
        cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonOverlapPhase1/test/expert/mergedPatterns.xml")),
)
            
###OMTF pattern maker configuration
process.omtfPatternMaker = cms.EDAnalyzer("OMTFPatternMaker",
                                          srcDTPh = cms.InputTag('simDtTriggerPrimitiveDigis'),
                                          srcDTTh = cms.InputTag('simDtTriggerPrimitiveDigis'),
                                          srcCSC = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
                                          srcRPC = cms.InputTag('simMuonRPCDigis'),                                              
                                          simTracksTag = cms.InputTag('g4SimHits'),
                                          makeGoldenPatterns = cms.bool(False),
                                          mergeXMLFiles = cms.bool(True),
                                          makeConnectionsMaps = cms.bool(False),                                      
                                          dropRPCPrimitives = cms.bool(False),                                    
                                          dropDTPrimitives = cms.bool(False),                                    
                                          dropCSCPrimitives = cms.bool(False),   
                                          ptCode = cms.int32(25),#this is old PAC pt scale.
                                          charge = cms.int32(1),
                                          )

process.p = cms.Path(process.omtfPatternMaker)
