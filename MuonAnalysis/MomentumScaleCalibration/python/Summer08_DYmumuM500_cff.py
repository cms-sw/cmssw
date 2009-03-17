import FWCore.ParameterSet.Config as cms

source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(
    "file:/data2/demattia/Data/DYmumuM500/Filter_DYmumuM500_1.root",
    "file:/data2/demattia/Data/DYmumuM500/Filter_DYmumuM500_2.root",
    "file:/data2/demattia/Data/DYmumuM500/Filter_DYmumuM500_3.root",
    "file:/data2/demattia/Data/DYmumuM500/Filter_DYmumuM500_4.root"
    )
                    )
