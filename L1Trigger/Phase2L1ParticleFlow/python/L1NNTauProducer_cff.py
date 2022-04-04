import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_cff import l1ctLayer1Barrel,l1ctLayer1HGCal,l1ctLayer1
from L1Trigger.Phase2L1ParticleFlow.l1pfProducer_cfi import l1pfProducer

#from L1Trigger.Phase2L1ParticleFlow.L1NNTauProducer_cfi import *

#L1NNTauProducerPuppi = L1NNTauProducer.clone(
#                                NNFileName      = cms.string("L1Trigger/Phase2L1ParticleFlow/data/tau_3layer_puppi.pb")
#                                )


L1NNTauProducerPuppi = cms.EDProducer("L1NNTauProducer",
                                      seedpt          = cms.double(10),
                                      conesize        = cms.double(0.4),
                                      tausize         = cms.double(0.1),
                                      maxtaus         = cms.int32(5),
                                      nparticles      = cms.int32(10),
                                      HW              = cms.bool(True),
                                      debug           = cms.bool(False),
                                      L1PFObjects     = cms.InputTag("l1ctLayer1:Puppi"), #1pfCandidates:Puppi"),#l1pfCandidates
                                      NNFileName      = cms.string("L1Trigger/Phase2L1ParticleFlow/data/tau_3layer_puppi.pb")
)

L1NNTauProducerPF = cms.EDProducer("L1NNTauProducer",
                                      seedpt          = cms.double(10),
                                      conesize        = cms.double(0.4),
                                      tausize         = cms.double(0.1),
                                      maxtaus         = cms.int32(5),
                                      nparticles      = cms.int32(10),
                                      HW              = cms.bool(True),
                                      debug           = cms.bool(False),
                                      L1PFObjects     = cms.InputTag("l1ctLayer1:PF"),#l1pfCandidates
                                      NNFileName      = cms.string("L1Trigger/Phase2L1ParticleFlow/data/tau_3layer.pb")
)


l1ctLayer1Barrel2Vtx = l1ctLayer1Barrel.clone()
l1ctLayer1Barrel2Vtx.nVtx = 2
l1ctLayer1HGCal2Vtx  = l1ctLayer1HGCal.clone()
l1ctLayer1HGCal2Vtx.nVtx = 2
l1ctLayer12Vtx       = l1ctLayer1.clone()
l1ctLayer12Vtx.pfProducers = cms.VInputTag(
    cms.InputTag("l1ctLayer1Barrel2Vtx"),
    cms.InputTag("l1ctLayer1HGCal2Vtx"),
    cms.InputTag("l1ctLayer1HGCalNoTK"),
    cms.InputTag("l1ctLayer1HF")
)
L1NNTauProducerPuppi2Vtx = L1NNTauProducerPuppi.clone()
L1NNTauProducerPuppi2Vtx.L1PFObjects =  cms.InputTag("l1ctLayer12Vtx:Puppi")
tau2VtxTaskHW = cms.Task(
    l1ctLayer1Barrel2Vtx,
    l1ctLayer1HGCal2Vtx,
    l1ctLayer12Vtx,
    L1NNTauProducerPuppi2Vtx
)

l1pfProducer2VtxSW         = l1pfProducer.clone()
l1pfProducer2VtxSW.nVtx    = 2
L1NNTauProducerPuppi2VtxSW = L1NNTauProducerPuppi.clone()
L1NNTauProducerPuppi2VtxSW.HW = False
L1NNTauProducerPuppi2VtxSW.L1PFObjects =  cms.InputTag("l1ctLayer12Vtx:Puppi")
tau2VtxTaskSW = cms.Task(
    l1pfProducer2VtxSW,
    L1NNTauProducerPuppi2VtxSW
)
