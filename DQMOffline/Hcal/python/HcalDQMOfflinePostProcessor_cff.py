import FWCore.ParameterSet.Config as cms

from DQMOffline.Hcal.HcalRecHitsDQMClient_cfi import *
from DQMOffline.Hcal.HcalNoiseRatesClient_cfi import *
from DQMOffline.Hcal.CaloTowersDQMClient_cfi import *

HcalDQMOfflinePostProcessor = cms.Sequence(hcalNoiseRatesClient*hcalRecHitsDQMClient*calotowersDQMClient)
#HcalDQMOfflinePostProcessor = cms.Sequence(hcalNoiseRatesClient*calotowersDQMClient)
