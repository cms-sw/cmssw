import FWCore.ParameterSet.Config as cms
# File: EcalHaloData_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module to build EcalHaloData Object and put into the event
# Date: Oct. 15, 2009

from RecoMET.METProducers.ecalHaloDataProducer_cfi import ecalHaloDataProducer as _ecalHaloDataProducer
EcalHaloData = _ecalHaloDataProducer.clone()


