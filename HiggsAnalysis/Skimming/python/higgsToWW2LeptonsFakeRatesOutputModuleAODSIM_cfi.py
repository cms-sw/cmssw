import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToWW2Leptons_FakeRatesEventContent_cff import *
higgsToWW2LeptonsFakeRatesOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    higgsToWW2LeptonsFakeRatesEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToWW2LeptonsFakeRatesAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('hww2lFakeRates_AODSIM.root')
)


