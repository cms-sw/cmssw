import FWCore.ParameterSet.Config as cms

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi
hbherecoMB = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi.hbheprereco.clone()

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
hfrecoMB = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi.hfreco.clone()

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi
horecoMB = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi.horeco.clone()


# switch off "Hcal ZS in reco":
hbherecoMB.dropZSmarkedPassed = cms.bool(False)
hfrecoMB.dropZSmarkedPassed = cms.bool(False)
horecoMB.dropZSmarkedPassed = cms.bool(False)

hcalLocalRecoSequenceNZS = cms.Sequence(hbherecoMB*hfrecoMB*horecoMB) 
