import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.idExternalRemap_cfi import *

eidLooseGSFixed = eidLoose.clone(
    candidateProducer = cms.InputTag('gedGsfElectronsGSFixed'),
    newToOldObjectMap = cms.InputTag('gedGsfElectronsGSFixed'),
)

eidRobustHighEnergyGSFixed = eidRobustHighEnergy.clone(
    candidateProducer = cms.InputTag('gedGsfElectronsGSFixed'),
    newToOldObjectMap = cms.InputTag('gedGsfElectronsGSFixed'),
)

eidRobustLooseGSFixed = eidRobustLoose.clone(
    candidateProducer = cms.InputTag('gedGsfElectronsGSFixed'),
    newToOldObjectMap = cms.InputTag('gedGsfElectronsGSFixed'),
)

eidRobustTightGSFixed = eidRobustTight.clone(
    candidateProducer = cms.InputTag('gedGsfElectronsGSFixed'),
    newToOldObjectMap = cms.InputTag('gedGsfElectronsGSFixed'),
)

eidTightGSFixed = eidTight.clone(
    candidateProducer = cms.InputTag('gedGsfElectronsGSFixed'),
    newToOldObjectMap = cms.InputTag('gedGsfElectronsGSFixed'),
)

ElectronIDExternalProducerRemapSequenceForPAT = cms.Sequence(
    eidLooseGSFixed + 
    eidRobustHighEnergyGSFixed + 
    eidRobustLooseGSFixed + 
    eidRobustTightGSFixed + 
    eidTightGSFixed
)

PhotonCutBasedIDLooseGSFixed = PhotonCutBasedIDLoose.clone(
    candidateProducer = cms.InputTag('gedPhotonsGSFixed'),
    newToOldObjectMap = cms.InputTag('gedPhotonsGSFixed'),
)

PhotonCutBasedIDLooseEMGSFixed = PhotonCutBasedIDLooseEM.clone(
    candidateProducer = cms.InputTag('gedPhotonsGSFixed'),
    newToOldObjectMap = cms.InputTag('gedPhotonsGSFixed'),
)

PhotonCutBasedIDTightGSFixed = PhotonCutBasedIDTight.clone(
    candidateProducer = cms.InputTag('gedPhotonsGSFixed'),
    newToOldObjectMap = cms.InputTag('gedPhotonsGSFixed'),
)

PhotonIDExternalProducerRemapSequenceForPAT = cms.Sequence(
    PhotonCutBasedIDLooseGSFixed +
    PhotonCutBasedIDLooseEMGSFixed +
    PhotonCutBasedIDTightGSFixed
)
