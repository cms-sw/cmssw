import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import *

from PhysicsTools.PatAlgos.recoLayer0.pfParticleSelectionForIso_cff import *
from PhysicsTools.PatAlgos.recoLayer0.pfLowPtElectronIsolationPAT_cff import *

sourceElectrons = cms.InputTag("lowPtGsfElectrons")
lowPtElectronMatch = electronMatch.clone(
   src = sourceElectrons,
   embedGsfTrack = cms.bool(False),
   embedGsfElectronCore = cms.bool(False),
)

patLowPtElectrons = patElectrons.clone(
   electronSource = sourceElectrons,
   pfElectronSource = cms.InputTag("FIXME"), #make PF crash if used
   pfCandidateMap = cms.InputTag("FIXME:electrons"),
   electronIDSources = cms.PSet(),
   genParticleMatch = cms.InputTag("lowPtElectronMatch"), ## Association between electrons and generator particles
   isoDeposits = cms.PSet(
      pfChargedHadrons = cms.InputTag("lowPtElPFIsoDepositChargedPAT" ),
      pfChargedAll = cms.InputTag("lowPtElPFIsoDepositChargedAllPAT" ),
      pfPUChargedHadrons = cms.InputTag("lowPtElPFIsoDepositPUPAT" ),
      pfNeutralHadrons = cms.InputTag("lowPtElPFIsoDepositNeutralPAT" ),
      pfPhotons = cms.InputTag("lowPtElPFIsoDepositGammaPAT" ),
      ),
   isolationValues = cms.PSet(
      pfChargedHadrons = cms.InputTag("lowPtElPFIsoValueCharged04PFIdPAT"),
      pfChargedAll = cms.InputTag("lowPtElPFIsoValueChargedAll04PFIdPAT"),
      pfPUChargedHadrons = cms.InputTag("lowPtElPFIsoValuePU04PFIdPAT" ),
      pfNeutralHadrons = cms.InputTag("lowPtElPFIsoValueNeutral04PFIdPAT" ),
      pfPhotons = cms.InputTag("lowPtElPFIsoValueGamma04PFIdPAT" ),
      ),
   isolationValuesNoPFId = cms.PSet(
      pfChargedHadrons = cms.InputTag("lowPtElPFIsoValueCharged04NoPFIdPAT"),
      pfChargedAll = cms.InputTag("lowPtElPFIsoValueChargedAll04NoPFIdPAT"),
      pfPUChargedHadrons = cms.InputTag("lowPtElPFIsoValuePU04NoPFIdPAT" ),
      pfNeutralHadrons = cms.InputTag("lowPtElPFIsoValueNeutral04NoPFIdPAT" ),
      pfPhotons = cms.InputTag("lowPtElPFIsoValueGamma04NoPFIdPAT" )
      )
) 

lowPtElPFIsoDepositChargedPAT.src = sourceElectrons
lowPtElPFIsoDepositChargedAllPAT.src = sourceElectrons
lowPtElPFIsoDepositNeutralPAT.src = sourceElectrons
lowPtElPFIsoDepositGammaPAT.src = sourceElectrons
lowPtElPFIsoDepositPUPAT.src = sourceElectrons

## for scheduled mode
makePatLowPtElectronsTask = cms.Task(
    pfParticleSelectionForIsoTask,
    pfLowPtElectronIsolationPATTask,
    lowPtElectronMatch,
    patLowPtElectrons
    )
makePatLowPtElectrons = cms.Sequence(makePatLowPtElectronsTask)
