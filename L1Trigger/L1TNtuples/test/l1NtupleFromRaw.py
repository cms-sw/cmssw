import FWCore.ParameterSet.Config as cms

# make ntuples from RAW (ie. remove RECO)

from L1TriggerDPG.L1Ntuples.l1Ntuple_cfg import *

process.p.remove(process.l1RecoTreeProducer)
process.p.remove(process.l1MuonRecoTreeProducer)

Stage1Layer2Emul=False  ## set to true to use new Stage1Layer2 Emulator results (not UCT2015)
if Stage1Layer2Emul:
    process.p.replace(process.l1NtupleProducer,process.l1NtupleProducerStage1Layer2)

# edit here
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.GlobalTag.globaltag = ''

readFiles.extend( [
        
    ] )

