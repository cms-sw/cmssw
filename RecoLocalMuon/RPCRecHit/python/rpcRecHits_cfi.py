
import FWCore.ParameterSet.Config as cms

rpcRecHits = cms.EDProducer("RPCRecHitProducer",
    recAlgoConfig = cms.PSet(

    ),
    recAlgo = cms.string('RPCRecHitStandardAlgo'),
    rpcDigiLabel = cms.InputTag('muonRPCDigis'),
    maskSource = cms.string('File'),
    maskvecfile = cms.FileInPath('RecoLocalMuon/RPCRecHit/data/RPCMaskVec.dat'),
    deadSource = cms.string('File'),
    deadvecfile = cms.FileInPath('RecoLocalMuon/RPCRecHit/data/RPCDeadVec.dat')
)

#disabling DIGI2RAW,RAW2DIGI chain for Phase2
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify(rpcRecHits, rpcDigiLabel = 'simMuonRPCDigis')

##
## Modify for the tau embedding methods cleaning step
##
from Configuration.ProcessModifiers.tau_embedding_cleaning_cff import tau_embedding_cleaning
from TauAnalysis.MCEmbeddingTools.Cleaning_RECO_cff import tau_embedding_rpcRecHits_cleaner
tau_embedding_cleaning.toReplaceWith(rpcRecHits, tau_embedding_rpcRecHits_cleaner)