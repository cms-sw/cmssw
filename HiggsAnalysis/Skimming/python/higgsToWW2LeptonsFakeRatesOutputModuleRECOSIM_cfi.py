import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToWW2Leptons_FakeRatesEventContent_cff import *
higgsToWW2LeptonsFakeRatesOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    RECOSIMEventContent,
    higgsToWW2LeptonsFakeRatesEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsToWW2LeptonsFakeRatesRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('hww2lFakeRates_RECOSIM.root')
)


