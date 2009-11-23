import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonAnalyzer_cfi import *


dqmElectronOfflineClient = cms.EDAnalyzer("ElectronOfflineClient",

    Verbosity = cms.untracked.int32(0),
    FinalStep = cms.string("AtLumiEnd"),
    InputFile = cms.string(""),
    OutputFile = cms.string(""),

)
