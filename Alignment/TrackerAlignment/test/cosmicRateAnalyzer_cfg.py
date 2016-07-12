# Implementation : Description of running this tool can be find  at :
# https://twiki.cern.ch/twiki/bin/view/CMS/TkAlCosmicsRateMonitoring


import FWCore.ParameterSet.Config as cms

import os

process = cms.Process("Demo")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load('Configuration.StandardSequences.MagneticField_cff') # B-field map
process.load('Configuration.Geometry.GeometryRecoDB_cff') # Ideal geometry and interface
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff") # Global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag.globaltag = GlobalTag("80X_dataRun2_Prompt_v2")

process.options = cms.untracked.PSet(
     SkipEvent= cms.untracked.vstring("ProductNotFound"), # make this exception fatal
  )
process.maxEvents=cms.untracked.PSet(input=cms.untracked.int32(-1)) 

import FWCore.Utilities.FileUtils as FileUtils
 
readFiles = cms.untracked.vstring()


readFiles = cms.untracked.vstring( FileUtils.loadListFromFile (os.environ['CMSSW_BASE']+'/src/Alignment/TrackerAlignment/test/'+'fileList.txt') )
process.source = cms.Source("PoolSource",
			   fileNames = readFiles,
			   )

process.TFileService = cms.Service("TFileService", fileName = cms.string("Cosmic_rate_tuple.root") )
process.demo = cms.EDAnalyzer("CosmicRateAnalyzer",
#				tracks = cms.InputTag("ctfWithMaterialTracksP5"),                # Track collection for prompt RECO Dataset
				tracksInputTag = cms.InputTag("ALCARECOTkAlCosmicsCTF0T"),		 # Track collection for stream and prompt ALCARECO Dataset
				muonsInputTag = cms.InputTag("muons1Leg"),		 # for muon Trigger timing information  
)

process.p = cms.Path(process.demo)


