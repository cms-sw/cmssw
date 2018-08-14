########################################
# Imports/Setup
########################################

import sys
import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(['root://xrootd.ba.infn.it//store/mc/RunIISpring16MiniAODv1/TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/0899BDA9-AE01-E611-A239-008CFA05EA2C.root']))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.OUT = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('test.root'),
                               outputCommands = cms.untracked.vstring(['drop *']))

process.task = cms.Task()
process.endpath= cms.EndPath(process.OUT, process.task)

# Let CMSSW take care of scheduling 
process.options = cms.untracked.PSet(     
    wantSummary = cms.untracked.bool(True)
)


########################################
# HEPTopTagger V2
########################################

from RecoJets.JetProducers.hepTopTaggerV2_cff import chsForHTT, hepTopTaggerV2

process.chsForHTT = chsForHTT
process.hepTopTaggerV2 = hepTopTaggerV2

process.task.add(process.chsForHTT, process.hepTopTaggerV2)

process.OUT.outputCommands.append("keep *_hepTopTaggerV2_*_Demo")
        



