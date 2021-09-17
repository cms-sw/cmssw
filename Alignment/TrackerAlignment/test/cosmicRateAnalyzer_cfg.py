# Implementation : Description of running this tool can be find  at :
# https://twiki.cern.ch/twiki/bin/view/CMS/TkAlCosmicsRateMonitoring

import FWCore.ParameterSet.Config as cms

import os

process = cms.Process("cosmicRateAnalyzer")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load('Configuration.StandardSequences.MagneticField_cff') # B-field map
process.load('Configuration.Geometry.GeometryRecoDB_cff') # Ideal geometry and interface
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff") # Global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

process.options = cms.untracked.PSet(
     SkipEvent= cms.untracked.vstring("ProductNotFound"), # make this exception fatal
  )
process.maxEvents=cms.untracked.PSet(input=cms.untracked.int32(10000)) 

import FWCore.Utilities.FileUtils as FileUtils
 
readFiles = cms.untracked.vstring()


readFiles = cms.untracked.vstring( FileUtils.loadListFromFile (os.environ['CMSSW_BASE']+'/src/Alignment/TrackerAlignment/test/'+'fileList.txt') )
process.source = cms.Source("PoolSource",
			   fileNames = readFiles,
			   )

process.TFileService = cms.Service("TFileService", fileName = cms.string("Cosmic_rate_tuple.root") )

process.load("Alignment.TrackerAlignment.cosmicRateAnalyzer_cfi")
process.p = cms.Path(process.cosmicRateAnalyzer)

