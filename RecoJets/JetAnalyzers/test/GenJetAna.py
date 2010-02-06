# PYTHON configuration file.
# Description:  Example of analysis pythia produced partons & jets
# Author: R. Harris
# Date:  28 - October - 2008
import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
#############   Define the source file ###############
process.load("RecoJets.JetAnalyzers.QCD_GenJets_cfi")
#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring(
#'file:/uscms_data/d1/rharris/CMSSW_2_1_8/src/Configuration/GenProduction/test/PYTHIA6_QCDpt_1000_1400_10TeV_GEN_100Kevts.root')
#)
#############   User analyzer  #####
# Jet Algorithm: which jets to use for making histograms and ana root tree.
# HistoFleName:  name of file which contains histograms and ana root tree.
# debug:         set true for printout each event
# NJets:         Currently only the value 2 is supported for dijet ana.
# eventsGen:     Number of events generated, or will be generated, per pt_hat bin.
# anaLevel:      Analysis level string which steers analysis choices
#               "PtHatOnly": only get PtHat and make PtHat histos. FAST.
#               "Jets":  do analysis of jets, but not partons.
#                        Input file only needs to have GenJets.
#               "all":   do analysis of everything and make histos and root tree
#                        Input file needs to have GenParticles as well as GenJets
#               "generating": prior modes all assumd we were running on file.
#                             This mode assumes we are generating pythia here.
#                            Analysis of everything, make histos and root tree
# xsecGen:       Generator cross section used if analysis level is "generating"
#
process.gen = cms.EDAnalyzer("GenJetAnaPythia",
    JetAlgorithm    = cms.string('sisCone7GenJets'),
    HistoFileName   = cms.string('TestQCDhistosGenJets.root'),
    debug           = cms.bool(False),
    NJets           = cms.int32(2),
    eventsGen       = cms.int32(800000),
    anaLevel        = cms.string('Jets'),
    xsecGen         = cms.double(0.0)
)
#############   Path       ###########################
process.p = cms.Path(process.gen)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

