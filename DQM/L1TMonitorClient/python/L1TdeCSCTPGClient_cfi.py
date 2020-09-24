import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tdeCSCTPGClient = DQMEDHarvester(
    "L1TdeCSCTPGClient",
    monitorDir = cms.string('L1TEMU/L1TdeCSCTPG'),
    alctVars = cms.vstring("quality", "wiregroup", "bx"),
    clctVars = cms.vstring("quality", "halfstrip", "bx",
                           "pattern", "bend", "quartstrip","eightstrip"),
    lctVars = cms.vstring("quality", "wiregroup", "halfstrip",
                          "bx", "pattern", "bend", "quartstrip","eightstrip"),
    alctNBin = cms.vuint32(16,116,20),
    alctMinBin = cms.vdouble(0.5,-0.5,-0.5),
    alctMaxBin = cms.vdouble(16.5,115.5,19.5),
    clctNBin = cms.vuint32(16),
    clctMinBin = cms.vdouble(0.5),
    clctMaxBin = cms.vdouble(16.5),
    lctNBin = cms.vuint32(16),
    lctMinBin = cms.vdouble(0.5),
    lctMaxBin = cms.vdouble(16.5),
)
