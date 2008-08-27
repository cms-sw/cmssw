# PYTHON configuration file for class: JetValidation
# Description:  Some Basic validation plots for jets.
# Author: K. Kousouris
# Date:  27 - August - 2008
import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#############   Define the source file ###############
#process.load("RecoJets.JetAnalyzers.RelValQCD_Pt_80_120_cfi")
process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_10TeV_v7/0000/044A7A03-BE6E-DD11-B3C0-000423D98B08.root'))
#############   Define the process ###################
process.validation = cms.EDAnalyzer("JetValidation",
    PtMin         = cms.double(5.),
    dRmatch       = cms.double(0.25),
    MCarlo        = cms.bool(True),
    calAlgo       = cms.string('iterativeCone5CaloJets'),
    genAlgo       = cms.string('iterativeCone5GenJets'),
    histoFileName = cms.string('JetValidation_ptMin5.root'),
    Njets         = cms.int32(100)
)
#############   Path       ###########################
process.p = cms.Path(process.validation)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

