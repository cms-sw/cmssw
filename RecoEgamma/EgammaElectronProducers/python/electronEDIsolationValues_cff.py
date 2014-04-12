import FWCore.ParameterSet.Config as cms

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

electronEDIsolationValuesSequence = cms.Sequence(
    elEDIsoValueCharged03+
    elEDIsoValueChargedAll03+
    elEDIsoValueGamma03+
    elEDIsoValueNeutral03+
    elEDIsoValuePU03+
############################## 
    elEDIsoValueCharged04+
    elEDIsoValueChargedAll04+
    elEDIsoValueGamma04+
    elEDIsoValueNeutral04+
    elEDIsoValuePU04
  )
