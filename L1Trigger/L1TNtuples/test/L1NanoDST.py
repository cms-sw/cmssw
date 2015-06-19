import FWCore.ParameterSet.Config as cms

fname = 'L1TreeL1Accept.root'


# cms process    
process = cms.Process("L1NTUPLE")
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/EventContent/EventContent_cff')

# analysis
process.load("L1TriggerDPG.L1Ntuples.l1NtupleProducerNano_cfi")

readFiles = cms.untracked.vstring()
readFiles.extend(['file:///tmp/guiducci/testL1Accept.root'])

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100000) )
process.source = cms.Source(
    "PoolSource",
    fileNames = readFiles
)
process.p = cms.Path(
    process.l1NtupleProducer
)
process.TFileService = cms.Service("TFileService",
    fileName = cms.string(fname)                                  
)
#Tue Sep 21 11:45:42 CEST 2010
