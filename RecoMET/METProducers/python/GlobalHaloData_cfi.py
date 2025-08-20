import FWCore.ParameterSet.Config as cms
# File: GlobalHaloData_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module to build GlobalHaloData Object and put into the event
# Date: Oct. 15, 2009

from RecoMET.METProducers.globalHaloDataProducer_cfi import globalHaloDataProducer as _globalHaloDataProducer
GlobalHaloData = _globalHaloDataProducer.clone()


