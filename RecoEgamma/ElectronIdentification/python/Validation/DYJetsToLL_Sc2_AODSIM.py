import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/mc/Spring14dr/DYJetsToLL_M-50_13TeV-madgraph-pythia8/AODSIM/PU20bx25_POSTLS170_V5-v1/00000/00350711-84CB-E311-BDBF-0025901D4C44.root',
       '/store/mc/Spring14dr/DYJetsToLL_M-50_13TeV-madgraph-pythia8/AODSIM/PU20bx25_POSTLS170_V5-v1/00000/0094D746-88CB-E311-A486-00266CFFA7A8.root',
       '/store/mc/Spring14dr/DYJetsToLL_M-50_13TeV-madgraph-pythia8/AODSIM/PU20bx25_POSTLS170_V5-v1/00000/00B3D5C6-C2CB-E311-9026-002481E0D480.root'
        ] )
