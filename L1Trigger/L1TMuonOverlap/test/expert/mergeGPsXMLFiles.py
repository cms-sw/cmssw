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


path = "job_3_pat/8_0_0_29_01_2016/" #Note: make a link of job_3_pat directory under CMSSW/src
patternsXMLFiles = cms.VPSet()
for ipt in xrange(4,32):
    patternsXMLFiles.append(cms.PSet(patternsXMLFile = cms.FileInPath(path+"SingleMu_"+str(ipt)+"_p/GPs.xml")))
    patternsXMLFiles.append(cms.PSet(patternsXMLFile = cms.FileInPath(path+"SingleMu_"+str(ipt)+"_m/GPs.xml")))
            
###OMTF pattern maker configuration
process.omtfPatternMaker = cms.EDAnalyzer("OMTFPatternMaker",
                                          srcDTPh = cms.InputTag('simDtTriggerPrimitiveDigis'),
                                          srcDTTh = cms.InputTag('simDtTriggerPrimitiveDigis'),
                                          srcCSC = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
                                          srcRPC = cms.InputTag('simMuonRPCDigis'),                                              
                                          g4SimTrackSrc = cms.InputTag('g4SimHits'),
                                          makeGoldenPatterns = cms.bool(False),
                                          mergeXMLFiles = cms.bool(True),
                                          makeConnectionsMaps = cms.bool(False),                                      
                                          dropRPCPrimitives = cms.bool(False),                                    
                                          dropDTPrimitives = cms.bool(False),                                    
                                          dropCSCPrimitives = cms.bool(False),   
                                          ptCode = cms.int32(25),#this is old PAC pt scale.
                                          charge = cms.int32(1),
                                          omtf = cms.PSet(
                                              configFromXML = cms.bool(True),   
                                              patternsXMLFiles = cms.VPSet(                                       
                                                  cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x00020007.xml")),
                                              ),
                                              configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x00020005.xml"),
                                          )
                                          )

process.omtfPatternMaker.omtf.patternsXMLFiles = patternsXMLFiles

process.p = cms.Path(process.omtfPatternMaker)
