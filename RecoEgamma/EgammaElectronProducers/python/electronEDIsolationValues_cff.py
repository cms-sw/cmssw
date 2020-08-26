import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.electronPFIsolationValues_cff import *

elEDIsoValueCharged03 = elPFIsoValueCharged03.clone(deposits = {0: dict(src ='elEDIsoDepositCharged')} )

elEDIsoValueChargedAll03 = elPFIsoValueChargedAll03.clone(deposits = {0: dict(src='elEDIsoDepositChargedAll')} )

elEDIsoValueGamma03 = elPFIsoValueGamma03.clone(deposits = {0: dict(src='elEDIsoDepositGamma')} )

elEDIsoValueNeutral03 = elPFIsoValueNeutral03.clone(deposits = {0: dict(src='elEDIsoDepositNeutral')} )

elEDIsoValuePU03  = elPFIsoValuePU03.clone(deposits = {0: dict(src = 'elEDIsoDepositPU')} )

elEDIsoValueCharged04 = elPFIsoValueCharged04.clone(deposits = {0: dict(src ='elEDIsoDepositCharged')} )

elEDIsoValueChargedAll04 = elPFIsoValueChargedAll04.clone(deposits = {0: dict(src='elEDIsoDepositChargedAll')} )

elEDIsoValueGamma04 = elPFIsoValueGamma04.clone(deposits = {0: dict(src='elEDIsoDepositGamma')} )

elEDIsoValueNeutral04 = elPFIsoValueNeutral04.clone(deposits = {0: dict(src='elEDIsoDepositNeutral')} )

elEDIsoValuePU04  = elPFIsoValuePU04.clone(deposits = {0: dict(src = 'elEDIsoDepositPU')} )

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
