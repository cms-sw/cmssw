import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

from CommonTools.RecoUtils.leptonFSRProducer_cfi import leptonFSRProducer
leptonFSRphotons = leptonFSRProducer.clone(
  packedPFCandidates = "packedPFCandidates",
  slimmedElectrons = "slimmedElectrons", #for footrprint veto
  muons = "linkedObjects:muons",
  electrons = "linkedObjects:electrons",
)

fsrTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("leptonFSRphotons"),
    cut = cms.string(""), #we should not filter on cross linked collections
    name = cms.string("FsrPhoton"),
    doc  = cms.string("Final state radiation photons emitted by muons or electrons"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(P3Vars,
        relIso03 = Var("userFloat('relIso03')",float,doc="relative isolation in a 0.3 cone without CHS"),
        dROverEt2 = Var("userFloat('dROverEt2')",float,doc="deltaR to associated muon divided by photon et2"),
        muonIdx = Var("?hasUserCand('associatedMuon')?userCand('associatedMuon').key():-1",int, doc="index of associated muon"),
        electronIdx = Var("?hasUserCand('associatedElectron')?userCand('associatedElectron').key():-1",int, doc="index of associated electron")
        )
    )

fsrTablesTask =  cms.Task(leptonFSRphotons,fsrTable)
