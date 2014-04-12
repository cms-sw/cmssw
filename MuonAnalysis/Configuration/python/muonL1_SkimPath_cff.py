import FWCore.ParameterSet.Config as cms

# this is just Juan's sequence to select l1 muons
from MuonAnalysis.Configuration.muonL1_cfi import *
muonL1Path = cms.Path(muonL1)

