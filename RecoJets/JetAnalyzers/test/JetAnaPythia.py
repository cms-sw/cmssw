# PYTHON configuration file.
# Description:  Example of analysis pythia produced partons & jets
# Author: R. Harris
# Date:  28 - October - 2008
import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#############   Define the source file ###############
process.load("RecoJets.JetAnalyzers.QCDgen_cfi")
#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring(
#'file:/uscms_data/d1/rharris/CMSSW_2_1_8/src/Configuration/GenProduction/test/PYTHIA6_QCDpt_1000_1400_10TeV_GEN_100Kevts.root')
#)
#############   User analyzer  #####
process.gen = cms.EDAnalyzer("GenJetAnaPythia",
    JetAlgorithm    = cms.string('sisCone7GenJets'),
    HistoFileName   = cms.string('QCDhistos.root'),
    debug           = cms.bool(False),
    NJets           = cms.int32(2),
    eventsGen       = cms.int32(100000),
    anaLevel        = cms.string('all'),
    xsecGen         = cms.double(0.0)
)
#############   Path       ###########################
process.p = cms.Path(process.gen)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

