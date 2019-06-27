import FWCore.ParameterSet.Config as cms

#==============================================================================
# Sequence to make final electrons.
# In the past, this was including the seeding, but this one is directly
# imported in the reco sequences since the integration with pflow.
#==============================================================================

from RecoEgamma.EgammaElectronProducers.gsfElectronCores_cfi import *
from RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi import *
gsfElectronTask = cms.Task(ecalDrivenGsfElectronCores,ecalDrivenGsfElectrons,gsfElectronCores,gsfElectrons)
gsfElectronSequence = cms.Sequence(gsfElectronTask)

gsfEcalDrivenElectronTask = cms.Task(ecalDrivenGsfElectronCores,ecalDrivenGsfElectrons)
gsfEcalDrivenElectronSequence = cms.Sequence(gsfEcalDrivenElectronTask)

_gsfEcalDrivenElectronTaskFromMultiCl = gsfEcalDrivenElectronTask.copy()
_gsfEcalDrivenElectronTaskFromMultiCl.add(cms.Task(ecalDrivenGsfElectronCoresFromMultiCl,ecalDrivenGsfElectronsFromMultiCl))
_gsfEcalDrivenElectronSequenceFromMultiCl = cms.Sequence(_gsfEcalDrivenElectronTaskFromMultiCl)

#gsfElectronMergingSequence = cms.Sequence(gsfElectronCores*gsfElectrons)

from RecoEgamma.EgammaElectronProducers.pfBasedElectronIso_cff import *
 
from CommonTools.ParticleFlow.pfParticleSelection_cff import *
from RecoEgamma.EgammaElectronProducers.electronPFIsolationDeposits_cff import *

elEDIsoDepositCharged=elPFIsoDepositCharged.clone()
elEDIsoDepositCharged.src="ecalDrivenGsfElectrons"

elEDIsoDepositChargedAll=elPFIsoDepositChargedAll.clone()
elEDIsoDepositChargedAll.src="ecalDrivenGsfElectrons"

elEDIsoDepositGamma=elPFIsoDepositGamma.clone()
elEDIsoDepositGamma.src="ecalDrivenGsfElectrons"

elEDIsoDepositNeutral=elPFIsoDepositNeutral.clone()
elEDIsoDepositNeutral.src="ecalDrivenGsfElectrons"

elEDIsoDepositPU=elPFIsoDepositPU.clone()
elEDIsoDepositPU.src="ecalDrivenGsfElectrons"

electronEDIsolationDepositsTask = cms.Task(
    elEDIsoDepositCharged,
    elEDIsoDepositChargedAll,
    elEDIsoDepositGamma,
    elEDIsoDepositNeutral,
    elEDIsoDepositPU
    )
electronEDIsolationDepositsSequence = cms.Sequence(electronEDIsolationDepositsTask)

from RecoEgamma.EgammaElectronProducers.electronPFIsolationValues_cff import *

elEDIsoValueCharged03 = elPFIsoValueCharged03.clone()
elEDIsoValueCharged03.deposits[0].src ='elEDIsoDepositCharged'

elEDIsoValueChargedAll03 = elPFIsoValueChargedAll03.clone()
elEDIsoValueChargedAll03.deposits[0].src='elEDIsoDepositChargedAll'

elEDIsoValueGamma03 = elPFIsoValueGamma03.clone()
elEDIsoValueGamma03.deposits[0].src='elEDIsoDepositGamma'

elEDIsoValueNeutral03 = elPFIsoValueNeutral03.clone()
elEDIsoValueNeutral03.deposits[0].src='elEDIsoDepositNeutral'

elEDIsoValuePU03  = elPFIsoValuePU03.clone()
elEDIsoValuePU03.deposits[0].src = 'elEDIsoDepositPU'

elEDIsoValueCharged04 = elPFIsoValueCharged04.clone()
elEDIsoValueCharged04.deposits[0].src ='elEDIsoDepositCharged'

elEDIsoValueChargedAll04 = elPFIsoValueChargedAll04.clone()
elEDIsoValueChargedAll04.deposits[0].src='elEDIsoDepositChargedAll'

elEDIsoValueGamma04 = elPFIsoValueGamma04.clone()
elEDIsoValueGamma04.deposits[0].src='elEDIsoDepositGamma'

elEDIsoValueNeutral04 = elPFIsoValueNeutral04.clone()
elEDIsoValueNeutral04.deposits[0].src='elEDIsoDepositNeutral'

elEDIsoValuePU04  = elPFIsoValuePU04.clone()
elEDIsoValuePU04.deposits[0].src = 'elEDIsoDepositPU'

electronEDIsolationValuesTask = cms.Task(
    elEDIsoValueCharged03,
    elEDIsoValueChargedAll03,
    elEDIsoValueGamma03,
    elEDIsoValueNeutral03,
    elEDIsoValuePU03,
############################## 
    elEDIsoValueCharged04,
    elEDIsoValueChargedAll04,
    elEDIsoValueGamma04,
    elEDIsoValueNeutral04,
    elEDIsoValuePU04
  )
electronEDIsolationValuesSequence = cms.Sequence(electronEDIsolationValuesTask)


edBasedElectronIsoTask = cms.Task(
    pfParticleSelectionTask,
    electronEDIsolationDepositsTask,
    electronEDIsolationValuesTask
    )
edBasedElectronIsoSequence = cms.Sequence(edBasedElectronIsoTask)

electronIsoTask = cms.Task(
        edBasedElectronIsoTask,
        pfBasedElectronIsoTask
     )
electronIsoSequence = cms.Sequence(electronIsoTask)

gsfElectronMergingTask = cms.Task(electronIsoTask,gsfElectronCores,gsfElectrons)
gsfElectronMergingSequence = cms.Sequence(gsfElectronMergingTask)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith(
  gsfEcalDrivenElectronTask, _gsfEcalDrivenElectronTaskFromMultiCl
)
