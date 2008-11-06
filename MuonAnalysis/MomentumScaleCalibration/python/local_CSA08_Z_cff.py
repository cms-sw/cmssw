import FWCore.ParameterSet.Config as cms

source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(
    "file:/data2/demattia/Data/CSA08/Z/Filter_Z_0.root",
    "file:/data2/demattia/Data/CSA08/Z/Filter_Z_10.root",
    "file:/data2/demattia/Data/CSA08/Z/Filter_Z_11.root",
    "file:/data2/demattia/Data/CSA08/Z/Filter_Z_1.root",
    "file:/data2/demattia/Data/CSA08/Z/Filter_Z_2.root",
    "file:/data2/demattia/Data/CSA08/Z/Filter_Z_3.root",
    "file:/data2/demattia/Data/CSA08/Z/Filter_Z_4.root",
    "file:/data2/demattia/Data/CSA08/Z/Filter_Z_5.root",
    "file:/data2/demattia/Data/CSA08/Z/Filter_Z_6.root",
    "file:/data2/demattia/Data/CSA08/Z/Filter_Z_7.root",
    "file:/data2/demattia/Data/CSA08/Z/Filter_Z_8.root",
    "file:/data2/demattia/Data/CSA08/Z/Filter_Z_9.root"
    )
                    )
