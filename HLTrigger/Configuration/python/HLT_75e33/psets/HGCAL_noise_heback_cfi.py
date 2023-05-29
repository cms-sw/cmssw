import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import HGCAL_noise_heback as _HGCAL_noise_heback
HGCAL_noise_heback = _HGCAL_noise_heback.clone()
