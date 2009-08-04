
import sys
import os
sys.path.append('.')
import dbs_discovery

import FWCore.ParameterSet.Config as cms

process = cms.Process("readelectrons")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source ("PoolSource",
    fileNames = cms.untracked.vstring(),
    secondaryFileNames = cms.untracked.vstring(),
)

process.source.fileNames.extend(dbs_discovery.search())

from RecoEgamma.Examples.mcAnalyzerStdBiningParameters_cff import *
from RecoEgamma.Examples.mcAnalyzerFineBiningParameters_cff import *

process.gsfElectronAnalysis = cms.EDAnalyzer("GsfElectronMCAnalyzer",
    electronCollection = cms.InputTag("gsfElectrons"),
    mcTruthCollection = cms.InputTag("generator"),
    outputFile = cms.string(os.environ['TEST_OUTPUT_FILE']),
    MaxPt = cms.double(100.0),
    DeltaR = cms.double(0.05),
    MaxAbsEta = cms.double(2.5),
    HistosConfigurationMC = cms.PSet(
    mcAnalyzerStdBiningParameters
    #mcAnalyzerFineBiningParameters
    )
)

process.p = cms.Path(process.gsfElectronAnalysis)


