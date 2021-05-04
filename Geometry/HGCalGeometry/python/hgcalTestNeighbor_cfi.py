import FWCore.ParameterSet.Config as cms

from Geometry.HGCalGeometry.hgcalEETestNeighbor_cfi import *

hgcalHEFTestNeighbor = hgcalEETestNeighbor.clone(
    detector = cms.string("HGCalHESiliconSensitive"))

hgcalHEBTestNeighbor = hgcalEETestNeighbor.clone(
    detector = cms.string("HGCalHEScintillatorSensitive"))
