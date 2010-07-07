import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("OLDREAD")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("file:"+sys.argv[2]))
