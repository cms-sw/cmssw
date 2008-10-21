import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi import *
hcalLocalRecoSequence = cms.Sequence(hbhereco+hfreco+horeco)

