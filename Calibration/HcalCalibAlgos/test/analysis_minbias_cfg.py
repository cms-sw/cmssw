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
GlobalTag.globaltag = "CRUZET4_V5P::All"

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
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/k/kodolova/CAF/CRUZET4/CRUZET4_V5P/ALCARECOHcalCalMinBiasCRUZET4_V5P_58553.root')
)


process.minbiasana = cms.EDAnalyzer("Analyzer_minbias",
    HistOutFile = cms.untracked.string('analysis_minbias_Full.root'),
    hbheInputMB = cms.InputTag("hbherecoMB"),
    hoInputMB = cms.InputTag("horecoMB"),
    hfInputMB = cms.InputTag("hfrecoMB"),
    hbheInputNoise = cms.InputTag("hbherecoNoise"),
    hoInputNoise = cms.InputTag("horecoNoise"),
    hfInputNoise = cms.InputTag("hfrecoNoise"),
    Recalib = cms.bool(False)
)

process.p1 = cms.Path(process.minbiasana)

