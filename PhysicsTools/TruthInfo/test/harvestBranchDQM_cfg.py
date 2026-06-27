# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Harvests a Branch-DQM file (output of validateBranchDQM_cfg.py) into efficiency
# plots via the truthGraphDQMHarvesting sequence (DQMGenericClient).

import FWCore.ParameterSet.Config as cms
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("inputFile", nargs='?', default="branch_dqm.root", metavar='FILE')
parser.add_argument('-o', "--out", default="branch_dqm_harvested.root")
args = parser.parse_args()
if '/' not in args.inputFile and ':' not in args.inputFile:
    args.inputFile = 'file:' + args.inputFile

process = cms.Process("BRANCHHARVEST")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMServices.Core.DQMStore_cfi")
process.load("PhysicsTools.TruthInfo.truthGraphDQMHarvester_cff")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))
process.source = cms.Source("DQMRootSource", fileNames=cms.untracked.vstring(args.inputFile))
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False), numberOfThreads=cms.untracked.uint32(1))

# Standard harvesting output: a legacy DQM_V0001_R*.root (TDirectory/TH1) written
# to the current directory under the workflow path below.
process.dqmSaver.convention = "Offline"
process.dqmSaver.workflow = "/Branch/Validation/HARVEST"
process.dqmSaver.saveByRun = cms.untracked.int32(1)

process.p = cms.Path(process.truthGraphDQMHarvesting)
process.e = cms.EndPath(process.dqmSaver)
