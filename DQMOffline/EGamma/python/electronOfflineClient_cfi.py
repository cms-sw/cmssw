import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonAnalyzer_cfi import *


dqmElectronOfflineClient = cms.EDAnalyzer("ElectronOfflineClient",

    Selection = cms.int32(1), # 0=All elec, 1=Etcut, 2=Iso, 3=eId, 4=T&P
    Verbosity = cms.untracked.int32(0),
    FinalStep = cms.string("AtRunEnd"),
    InputFile = cms.string(""),
    OutputFile = cms.string(""),
    InputFolderName = cms.string(""),
    OutputFolderName = cms.string("")

)
