#
# Example config for running HcalNoiseAnalyzer
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("HcalNoiseAnalyzerTest")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.source = cms.Source("PoolSource",
    # replace 'pythia_reco.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:pythia_reco.root'
    )
)

# Configure HBHENoiseFilterResultProducer
from CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi import *
HBHENoiseFilterResultProducer.jetlabel = cms.InputTag('ak5PFJets')

# Configure HcalNoiseAnalyzer
from CalibCalorimetry.HcalNoiseAnalyzer.hcalnoiseanalyzer_cfi import *

process.hbhefilter = HBHENoiseFilterResultProducer
process.noiseanalyzer = hcalNoiseAnalyzer

process.TFileService = cms.Service("TFileService",
   fileName = cms.string("NoiseTree.root")
)

process.p = cms.Path(
    process.hbhefilter *
    process.noiseanalyzer
)
