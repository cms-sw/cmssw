#
# This configuration file collects the data needed to calibrate
# FFTJet pule-up estimator for CaloJets
#
# Replace strings INPUTFILE and OUTPUTFILE in the configuration below
#
# I. Volobouev, April 27 2011
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("FFTJetTest")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("OUTPUTFILE")
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:INPUTFILE'
    )
)

from RecoJets.FFTJetProducers.fftjetpileupprocessor_caloprod_cfi import *
from RecoJets.FFTJetProducers.fftjetpileupestimator_calo_cfi import *
from RecoJets.JetAnalyzers.fftjetpileupanalyzer_cfi import *

fftjet_pileup_analyzer.summaryLabel = cms.InputTag(
    "pileupestimator", "FFTJetPileupEstimateCalo")

process.pileupprocessor = fftjet_pileup_processor_calo
process.pileupestimator = fftjet_pileup_estimator_calo
process.pileupanalyzer = fftjet_pileup_analyzer

process.p = cms.Path(process.pileupprocessor*process.pileupestimator*process.pileupanalyzer)
# foo bar baz
# JsvgJZx7YKoKN
