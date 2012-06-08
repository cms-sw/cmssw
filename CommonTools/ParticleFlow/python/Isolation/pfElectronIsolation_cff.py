import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.electronPFIsolationDeposits_cff import *
from RecoParticleFlow.PFProducer.electronPFIsolationValues_cff import *

elPFIsoValueGamma03.deposits[0].vetos = cms.vstring('Threshold(0.0)')
elPFIsoValueNeutral03.deposits[0].vetos = cms.vstring('Threshold(0.0)')
elPFIsoValuePU03.deposits[0].vetos = cms.vstring('Threshold(0.0)')
elPFIsoValueGamma04.deposits[0].vetos = cms.vstring('Threshold(0.0)')
elPFIsoValueNeutral04.deposits[0].vetos = cms.vstring('Threshold(0.0)')
elPFIsoValuePU04.deposits[0].vetos = cms.vstring('Threshold(0.0)')

pfElectronIsolationSequence = cms.Sequence(
    electronPFIsolationDepositsSequence +
    electronPFIsolationValuesSequence
    )

