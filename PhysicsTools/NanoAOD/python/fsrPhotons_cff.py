import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer

from CommonTools.RecoUtils.leptonFSRProducer_cfi import leptonFSRProducer
leptonFSRphotons = leptonFSRProducer.clone(
  packedPFCandidates = "packedPFCandidates",
  slimmedElectrons = "slimmedElectrons", #for footrprint veto
  muons = "linkedObjects:muons",
  electrons = "linkedObjects:electrons",
)

fsrTable = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("leptonFSRphotons"),
    name = cms.string("FsrPhoton"),
    doc  = cms.string("Final state radiation photons emitted by muons or electrons"),
    variables = cms.PSet(P3Vars,
        relIso03 = Var("userFloat('relIso03')",float,doc="relative isolation in a 0.3 cone without CHS"),
        dROverEt2 = Var("userFloat('dROverEt2')",float,doc="deltaR to associated muon divided by photon et2"),
        muonIdx = Var("?hasUserCand('associatedMuon')?userCand('associatedMuon').key():-1", "int16", doc="index of associated muon"),
        electronIdx = Var("?hasUserCand('associatedElectron')?userCand('associatedElectron').key():-1", "int16", doc="index of associated electron")
        )
    )

fsrTablesTask =  cms.Task(leptonFSRphotons,fsrTable)
