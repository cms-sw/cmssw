import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.coreTools import *

process = cms.Process("Demo")

process.load("RecoLocalCalo/EcalRecAlgos/EcalSeverityLevelESProducer_cfi")
process.load("Calibration.IsolatedParticles.studyHLT_cfi")
process.load('Configuration/Geometry/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('TrackingTools/TrackAssociator/DetIdAssociatorESProducer_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")


process.GlobalTag.globaltag = 'START53_V7::All' ## 2012 AODSIM  ##for MC
#process.GlobalTag.globaltag = 'FT_R_53_V6::All' ## July15ReReco Run2012A & B
#process.GlobalTag.globaltag = 'FT_53_V6C_AN4' ## July13ReReco Run2012A & B in 53X

################# CommandLine Parsing
#import os
#import sys
#import FWCore.ParameterSet.VarParsing as VarParsing
# setup 'standard'  options
#options = VarParsing.VarParsing ('standard')
#options.register ( "TrigNames",
#                 [],    
#                   VarParsing.VarParsing.multiplicity.list, # singleton or list
#                   VarParsing.VarParsing.varType.string,          # string, int, or float
#                   "HLT names")
#options.parseArguments()
#print options.TrigNames

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
#      '/store/data/Run2012A/MinimumBias/RECO/13Jul2012-v1/00000/001767E2-FFCF-E111-BF8A-003048FFD76E.root',
#     '/store/data/Run2012A/MinimumBias/RECO/13Jul2012-v1/00001/CAFCD70A-6BD0-E111-B8AD-003048678B1A.root',
#    '/store/data/Run2012A/MinimumBias/RECO/13Jul2012-v1/00001/BAECADE6-79D0-E111-B4A4-00261894382A.root',
#   '/store/data/Run2012A/MinimumBias/RECO/13Jul2012-v1/00001/BA276120-6BD0-E111-A2D8-00304867C16A.root',
#  '/store/data/Run2012A/MinimumBias/RECO/13Jul2012-v1/00000/1E2DE509-62D0-E111-8A5C-0026189437F9.root'
  '/store/mc/Summer12_DR53X/MinBias_TuneZ2star_8TeV-pythia6/AODSIM/PU_S10_START53_V7A-v1/0001/2A30B9E4-69DD-E111-B749-003048C66180.root',  ##MC
 '/store/mc/Summer12_DR53X/MinBias_TuneZ2star_8TeV-pythia6/AODSIM/PU_S10_START53_V7A-v1/0001/2A567389-11D9-E111-BE9C-0030487E52A1.root'    ##MC
    )
                            )

process.StudyHLT.Verbosity = 11
process.StudyHLT.IsItAOD   = True
#process.StudyHLT.Triggers  = options.TrigNames
#process.StudyHLT.Triggers  = ["PixelTracks_Multiplicity"]


process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('StudyHLT.root')
                                   )

process.p = cms.Path(process.StudyHLT)
