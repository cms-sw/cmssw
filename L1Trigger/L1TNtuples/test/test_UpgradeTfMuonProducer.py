
import FWCore.ParameterSet.Config as cms

process = cms.Process("L1NTUPLE")

# input file

process.source = cms.Source ("PoolSource",
   fileNames = cms.untracked.vstring(
        'file:///afs/cern.ch/work/g/gflouris/public/l1tbmtf_data_Run271820.root'),
    #eventsToProcess=cms.untracked.VEventRange('267878:19:46902-267878:19:46902')
   )



# N events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

# output file
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('L1Tree.root')
)

# producer under test
process.load("L1Trigger.L1TNtuples.l1UpgradeTfMuonTree_cfi")
process.l1UpgradeTfMuonTree.bmtfMuonToken = "BMTFStage2Digis:BMTF"
process.l1UpgradeTfMuonTree.bmtfInputPhMuonToken = "BMTFStage2Digis:PhiDigis"
process.l1UpgradeTfMuonTree.bmtfInputThMuonToken = "BMTFStage2Digis:TheDigis"

process.l1UpgradeTfMuonTree.omtfMuonToken = "none"
process.l1UpgradeTfMuonTree.emtfMuonToken = "none"

process.p = cms.Path(
  process.l1UpgradeTfMuonTree
)

