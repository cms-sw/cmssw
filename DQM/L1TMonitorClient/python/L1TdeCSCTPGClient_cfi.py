import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tdeCSCTPGClient = DQMEDHarvester(
    "L1TdeCSCTPGClient",
    monitorDir = cms.string('L1TEMU/L1TdeCSCTPG'),
    chambers = cms.vstring("ME1a", "ME1b", "ME12", "ME13", "ME21", "ME22",
                           "ME31", "ME32", "ME41", "ME42"),
    alctVars = cms.vstring("quality", "wiregroup", "bx"),
    clctVars = cms.vstring("quality", "halfstrip", "bx",
                           "pattern", "bend", "quartstrip","eightstrip"),
    lctVars = cms.vstring("quality", "wiregroup", "halfstrip",
                          "bx", "pattern", "bend", "quartstrip","eightstrip"),
    alctNBin = cms.vuint32(6,116,20),
    alctMinBin = cms.vdouble(-0.5,-0.5,-0.5),
    alctMaxBin = cms.vdouble(5.5,115.5,19.5),
    clctNBin = cms.vuint32(16, 240, 20, 10, 2, 2, 2),
    clctMinBin = cms.vdouble(-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5),
    clctMaxBin = cms.vdouble(-5.5, 239.5, 19.5, 9.5, 1.5, 1.5, 1.5),
    lctNBin = cms.vuint32(116, 16, 240, 20, 10, 2, 2, 2),
    lctMinBin = cms.vdouble(-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5),
    lctMaxBin = cms.vdouble(115.5, -5.5, 239.5, 19.5, 9.5, 1.5, 1.5, 1.5),
)
