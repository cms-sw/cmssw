import FWCore.ParameterSet.Config as cms
# File: HcalHaloData_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module to build HcalHaloData Object and put into the event
# Date: Oct. 15, 2009

from RecoMET.METProducers.hcalHaloDataProducer_cfi import hcalHaloDataProducer as _hcalHaloDataProducer
HcalHaloData = _hcalHaloDataProducer.clone()


