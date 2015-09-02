import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/mc/Spring14dr/DYJetsToLL_M-50_13TeV-madgraph-pythia8/AODSIM/PU_S14_POSTLS170_V6-v1/00000/0015495C-66CC-E311-8A39-00266CF33100.root',
       '/store/mc/Spring14dr/DYJetsToLL_M-50_13TeV-madgraph-pythia8/AODSIM/PU_S14_POSTLS170_V6-v1/00000/007F8470-ADCB-E311-A1BB-003048C6931E.root',
       '/store/mc/Spring14dr/DYJetsToLL_M-50_13TeV-madgraph-pythia8/AODSIM/PU_S14_POSTLS170_V6-v1/00000/00BBF612-84CC-E311-AB94-00266CF270A8.root'
        ] )


