import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.nano_cff import *

##for gen and trigger muon
from PhysicsTools.BPHNano.pverticesBPH_cff import *
from PhysicsTools.BPHNano.genparticlesBPH_cff import *
from PhysicsTools.BPHNano.particlelevelBPH_cff import *

## BPH collections
from PhysicsTools.BPHNano.muons_cff import *
from PhysicsTools.BPHNano.MuMu_cff import *
from PhysicsTools.BPHNano.tracks_cff import *
from PhysicsTools.BPHNano.DiTrack_cff import *
from PhysicsTools.BPHNano.V0_cff import *
from PhysicsTools.BPHNano.BToKLL_cff import *
from PhysicsTools.BPHNano.BToTrkTrkLL_cff import *
from PhysicsTools.BPHNano.BToV0LL_cff import *

def nanoAOD_customizeMC(process):
    process.load('PhysicsTools.BPHNano.particlelevelBPH_cff')
    process.load('PhysicsTools.BPHNano.genparticlesBPH_cff')
    process.nanoSequence = cms.Sequence(process.nanoSequence +process.particleLevelBPHSequence + process.genParticleBPHSequence+ process.genParticleBPHTables )
    return process



def nanoAOD_customizeMuonBPH(process):
    process.load('PhysicsTools.BPHNano.muons_cff')
    process.nanoSequence = cms.Sequence( process.nanoSequence + process.muonBPHSequence + process.muonBPHTables)
    return process


def nanoAOD_customizeDiMuonBPH(process):
    process.load('PhysicsTools.BPHNano.MuMu_cff')
    process.nanoSequence = cms.Sequence( process.nanoSequence + MuMuSequence + MuMuTables)
    return process



def nanoAOD_customizeTrackBPH(process):
    process.load('PhysicsTools.BPHNano.tracks_cff')    
    process.nanoSequence = cms.Sequence( process.nanoSequence + tracksBPHSequence + tracksBPHTables)
    return process


def nanoAOD_customizeBToKLL(process):
    process.load('PhysicsTools.BPHNano.BToKLL_cff')
    process.nanoSequence = cms.Sequence( process.nanoSequence + BToKMuMuSequence + BToKMuMuTables)
    return process



def nanoAOD_customizeBToTrkTrkLL(process):
    process.load('PhysicsTools.BPHNano.DiTrack_cff')    
    process.load('PhysicsTools.BPHNano.BToTrkTrkLL_cff')    
    process.nanoSequence = cms.Sequence( process.nanoSequence + DiTrackSequence + DiTrackTables+ BToTrkTrkMuMuSequence + BToTrkTrkMuMuTables  )
    return process




def nanoAOD_customizeBToKshortLL(process):
    process.load('PhysicsTools.BPHNano.V0_cff')
    process.load('PhysicsTools.BPHNano.BToV0LL_cff') 
    process.nanoSequenceMC = cms.Sequence( process.nanoSequence+ KshortToPiPiSequenceMC + KshortToPiPiTablesMC + BToKshortMuMuSequence + BToKshortMuMuTables  )
    process.nanoSequence = cms.Sequence( process.nanoSequence+ KshortToPiPiSequence + KshortToPiPiTables + BToKshortMuMuSequence + BToKshortMuMuTables  )
    return process

def nanoAOD_customizeLambdabToLambdaLL(process):
    process.load('PhysicsTools.BPHNano.V0_cff')
    process.load('PhysicsTools.BPHNano.BToV0LL_cff')
    process.nanoSequenceMC = cms.Sequence( process.nanoSequence+ LambdaToProtonPiSequenceMC + LambdaToProtonPiTablesMC + LambdabToLambdaMuMuSequence + LambdabToLambdaMuMuTables  )
    process.nanoSequence = cms.Sequence( process.nanoSequence+ LambdaToProtonPiSequence + LambdaToProtonPiTables + LambdabToLambdaMuMuSequence + LambdabToLambdaMuMuTables  )
    return process




def nanoAOD_customizeBPH(process):
    process.load('PhysicsTools.BPHNano.genparticlesBPH_cff')
    process.load('PhysicsTools.BPHNano.muons_cff')
    process.load('PhysicsTools.BPHNano.MuMu_cff')    
    process.load('PhysicsTools.BPHNano.tracks_cff')
    process.load('PhysicsTools.BPHNano.BToKLL_cff')    
    process.load('PhysicsTools.BPHNano.DiTrack_cff')
    process.load('PhysicsTools.BPHNano.BToTrkTrkLL_cff')
    process.load('PhysicsTools.BPHNano.V0_cff')
    process.load('PhysicsTools.BPHNano.BToV0LL_cff')      
    process.nanoSequenceMC = cms.Sequence(process.nanoSequenceMC +particleLevelBPHSequence + genParticleBPHSequence+ genParticleBPHTables + muonBPHSequenceMC + muonBPHTablesMC + MuMuSequence + MuMuTables + tracksBPHSequenceMC + tracksBPHTablesMC + BToKMuMuSequence + BToKMuMuTables + DiTrackSequence + DiTrackTables + BToTrkTrkMuMuSequence + BToTrkTrkMuMuTables + KshortToPiPiSequenceMC + KshortToPiPiTablesMC + BToKshortMuMuSequence + BToKshortMuMuTables +  LambdaToProtonPiSequenceMC + LambdaToProtonPiTablesMC + LambdabToLambdaMuMuSequence + LambdabToLambdaMuMuTables)
    process.nanoSequence = cms.Sequence(process.nanoSequence + muonBPHSequence + muonBPHTables + MuMuSequence + MuMuTables + tracksBPHSequence + tracksBPHTables + BToKMuMuSequence + BToKMuMuTables + DiTrackSequence + DiTrackTables + BToTrkTrkMuMuSequence + BToTrkTrkMuMuTables + KshortToPiPiSequence + KshortToPiPiTables + BToKshortMuMuSequence + BToKshortMuMuTables +  LambdaToProtonPiSequence + LambdaToProtonPiTables + LambdabToLambdaMuMuSequence + LambdabToLambdaMuMuTables)
    return process


