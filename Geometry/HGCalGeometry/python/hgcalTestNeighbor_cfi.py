import FWCore.ParameterSet.Config as cms

from Geometry.HGCalGeometry.hgcalEETestNeighbor_cfi import *

hgcalHEFTestNeighbor = hgcalEETestNeighbor.clone(
    detector = cms.string("HGCalHESiliconSensitive"))

hgcalHEBTestNeighbor = hgcalEETestNeighbor.clone(
    detector = cms.string("HCal"))

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

phase2_hgcalV9.toModify(hgcalHEBTestNeighbor,
                        detector = cms.string("HGCalHEScintillatorSensitive")
                        )
