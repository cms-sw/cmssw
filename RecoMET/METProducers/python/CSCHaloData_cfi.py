import FWCore.ParameterSet.Config as cms

# File: CSCHaloData_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module to build CSCHaloData and put into the event
# Date: Oct. 15, 2009

from RecoMET.METProducers.cscHaloDataProducer_cfi import cscHaloDataProducer as _cscHaloDataProducer
CSCHaloData = _cscHaloDataProducer.clone()
