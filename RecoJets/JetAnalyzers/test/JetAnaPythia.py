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
#process.load("RecoJets.JetAnalyzers.QCDgen_cfi")
process.source = cms.Source("PoolSource",
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
    fileNames = cms.untracked.vstring(
'file:/uscms_data/d2/rharris/CMSSW_3_3_2/src/PYTHIA6_QCDpt_120_inf_900GeV_cff_py_GEN.root')
)
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
# xsecGen:       Generator cross section.
#                   Just one value if analysis level is "generating"
#                   Multiple values if analysis level is "all"
# ptHatEdges:    Edges of ptHatBins that correspond to xsecGen values above
#
#
process.gen = cms.EDAnalyzer("GenJetAnaPythia",
    JetAlgorithm    = cms.string('ak7GenJets'),
    HistoFileName   = cms.string('TestQCDhistos.root'),
    debug           = cms.bool(False),
    NJets           = cms.int32(2),
    eventsGen       = cms.int32(100000),
    anaLevel        = cms.string('all'),
    xsecGen         = cms.vdouble(3.439E+10, 1.811E+07, 5.469E+06, 7.391E+05, 4.321E+04, 2.200E+03, 1.080E+02),
    ptHatEdges      = cms.vdouble(0, 15, 20, 30, 50, 80, 120, 9999)
)
#############   Path       ###########################
process.p = cms.Path(process.gen)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

