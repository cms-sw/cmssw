import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *

process = cms.Process("RECO3")

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

es_ascii2 = cms.ESSource("HcalTextCalibrations",
                         appendToDataLabel = cms.string('recalibrate'),
                         input = cms.VPSet(
			    cms.PSet(
			        object = cms.string('RespCorrs'),
				file = cms.FileInPath('coef_without_noise_10mln_pure_1pb.txt')
			    )
			 )
) 

es_prefer_es_ascii2 = cms.ESPrefer("HcalTextCalibrations","es_ascii2")
			 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/kodolova/FC999068-DB60-DD11-9694-001A92971B16.root')
)


process.minbiasana = cms.EDFilter("Analyzer_minbias",
    HistOutFile = cms.untracked.string('analysis_minbias.root'),
    nameProd = cms.untracked.string('MinProd'),
    hbheInput = cms.InputTag("hbhereco"),
    hoInput = cms.InputTag("horeco"),
    hfInput = cms.InputTag("hfreco"),
    hbheCut = cms.double(-20000.),
    hoCut = cms.double(-20000.),
    hfCut = cms.double(-20000.),
    useMC = cms.bool(False),
    Recalib = cms.bool(False)
)

process.p1 = cms.Path(process.minbiasana)

