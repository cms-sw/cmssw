#
# This configuration file collects the data needed to calibrate
# FFTJet pule-up estimator for PFJets
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

from RecoJets.FFTJetProducers.fftjetpileupprocessor_pfprod_cfi import *
from RecoJets.FFTJetProducers.fftjetpileupestimator_pf_cfi import *
from RecoJets.JetAnalyzers.fftjetpileupanalyzer_cfi import *

fftjet_pileup_analyzer.summaryLabel = cms.InputTag(
    "pileupestimator", "FFTJetPileupEstimatePF")
fftjet_pileup_analyzer.fastJetRhoLabel = cms.InputTag("mykt6PFJets", "rho")
fftjet_pileup_analyzer.fastJetSigmaLabel = cms.InputTag("mykt6PFJets", "sigma")
fftjet_pileup_analyzer.collectFastJetRho = cms.bool(True)

process.pileupprocessor = fftjet_pileup_processor_pf
process.pileupestimator = fftjet_pileup_estimator_pf
process.pileupanalyzer = fftjet_pileup_analyzer

# Configure FastJet rho reconstruction
from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets
mykt6PFJets = kt4PFJets.clone( rParam = 0.6 )
mykt6PFJets.doRhoFastjet = True
mykt6PFJets.doAreaFastjet = True
mykt6PFJets.voronoiRfact = 0.9
process.mykt6PFJets = mykt6PFJets

process.p = cms.Path(process.mykt6PFJets*process.pileupprocessor*process.pileupestimator*process.pileupanalyzer)
