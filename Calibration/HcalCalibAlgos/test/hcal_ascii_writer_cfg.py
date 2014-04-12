import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *

process = cms.Process("RECO4")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

GlobalTag.DBParameters.connectionTimeOut=60
GlobalTag.DBParameters.authenticationPath="/afs/cern.ch/cms/DB/conddb"
GlobalTag.globaltag = "STARTUP_V4::All"
			 
process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(1)
)


process.asciiwriter = cms.EDAnalyzer("HcalConstantsASCIIWriter",
    fileInput = cms.string('minbias_calib_output'),
    fileOutput = cms.string('minbias_calib_output_mult')
)

process.p1 = cms.Path(process.asciiwriter)

