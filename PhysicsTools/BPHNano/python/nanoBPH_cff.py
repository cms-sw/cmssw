from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.globals_cff import *
from PhysicsTools.NanoAOD.nano_cff import *
from PhysicsTools.NanoAOD.vertices_cff import *
from PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff import *
from PhysicsTools.NanoAOD.triggerObjects_cff import *


##for gen and trigger muon
from PhysicsTools.BPHNano.pverticesBPH_cff import *
from PhysicsTools.BPHNano.genparticlesBPH_cff import *
from PhysicsTools.BPHNano.particlelevelBPH_cff import *

## BPH collections
from PhysicsTools.BPHNano.muons_cff import *
from PhysicsTools.BPHNano.MuMu_cff import *
from PhysicsTools.BPHNano.tracks_cff import *
from PhysicsTools.BPHNano.DiTrack_cff import *
from PhysicsTools.BPHNano.KshortToPiPi_cff import *
from PhysicsTools.BPHNano.BToKLL_cff import *
from PhysicsTools.BPHNano.BToTrkTrkLL_cff import *
from PhysicsTools.BPHNano.BToKshortLL_cff import *


vertexTable.svSrc = cms.InputTag("slimmedSecondaryVertices")



nanoSequence = cms.Sequence(nanoMetadata + 
                            cms.Sequence(vertexTask) +
                            cms.Sequence(globalTablesTask)+ 
                            cms.Sequence(vertexTablesTask) +
                            cms.Sequence(pVertexTable)+
                            cms.Sequence(nanoSequenceCommon)                           
                          )



def nanoAOD_customizeMC(process):
    process.nanoSequence = cms.Sequence(process.nanoSequence +particleLevelBPHSequence + genParticleBPHSequence+ genParticleBPHTables )
    return process



def nanoAOD_customizeMuonBPH(process,isMC):
    if isMC:
       process.nanoSequence = cms.Sequence( process.nanoSequence + muonBPHSequenceMC + muonBPHTablesMC)
    else:
       process.nanoSequence = cms.Sequence( process.nanoSequence + muonBPHSequence + countTrgMuons + muonBPHTables)
    return process



def nanoAOD_customizeDiMuonBPH(process, isMC):
    if isMC:
       process.nanoSequence = cms.Sequence( process.nanoSequence + MuMuSequence + MuMuTables )
    else:
       process.nanoSequence = cms.Sequence( process.nanoSequence + MuMuSequence + CountDiMuonBPH + MuMuTables)
    return process



def nanoAOD_customizeTrackBPH(process,isMC):
    if isMC:
       process.nanoSequence =  cms.Sequence( process.nanoSequence + tracksBPHSequenceMC + tracksBPHTablesMC)
    else:
       process.nanoSequence = cms.Sequence( process.nanoSequence + tracksBPHSequence + tracksBPHTables)
    return process



def nanoAOD_customizeBToKLL(process,isMC):
    if isMC:
       process.nanoSequence = cms.Sequence( process.nanoSequence + BToKMuMuSequence + BToKMuMuTables  )
    else:
       process.nanoSequence = cms.Sequence( process.nanoSequence + BToKMuMuSequence +CountBToKmumu + BToKMuMuTables)
    return process



def nanoAOD_customizeBToTrkTrkLL(process,isMC):
    if isMC:
       process.nanoSequence = cms.Sequence( process.nanoSequence + DiTrackSequence + DiTrackTables + BToTrkTrkMuMuSequence + BToTrkTrkMuMuTables  )
    else:
       process.nanoSequence = cms.Sequence( process.nanoSequence + DiTrackSequence + CountDiTrack + DiTrackTables+ BToTrkTrkMuMuSequence + BToTrkTrkMuMuTables  )
    return process




def nanoAOD_customizeBToKshortLL(process, isMC):
    if isMC:
       process.nanoSequence = cms.Sequence( process.nanoSequence+ KshortToPiPiSequenceMC + KshortToPiPiTablesMC + BToKshortMuMuSequence + BToKshortMuMuTables  )
    else:
       process.nanoSequence = cms.Sequence( process.nanoSequence+ KshortToPiPiSequence + CountKshortToPiPi+ KshortToPiPiTables + BToKshortMuMuSequence + CountBToKshortMuMu +BToKshortMuMuTables  )
    return process




def nanoAOD_customizeBToXLL(process,isMC):
    if isMC:
       process.nanoSequence = cms.Sequence( process.nanoSequence + BToKMuMuSequence + BToKMuMuTables + KshortToPiPiSequenceMC + KshortToPiPiTablesMC + BToKshortMuMuSequence + BToKshortMuMuTables + DiTrackSequence + DiTrackTables+ BToTrkTrkMuMuSequence + BToTrkTrkMuMuTables  )
    else:
       process.nanoSequence = cms.Sequence( process.nanoSequence + BToKMuMuSequence + BToKMuMuTables + KshortToPiPiSequence + KshortToPiPiTables + BToKshortMuMuSequence +BToKshortMuMuTables + DiTrackSequence + DiTrackSequence +DiTrackTables+ BToTrkTrkMuMuSequence + BToTrkTrkMuMuTables )
    return process


