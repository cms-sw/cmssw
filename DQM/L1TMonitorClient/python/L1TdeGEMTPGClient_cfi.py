import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tdeGEMTPGClient = DQMEDHarvester(
    "L1TdeGEMTPGClient",
    monitorDir = cms.string('L1TEMU/L1TdeGEMTPG'),
    chambers = cms.vstring("GE11"),
    clusterVars = cms.vstring("size", "pad", "bx"),
    clusterNBin = cms.vuint32(20,384,10),
    clusterMinBin = cms.vdouble(-0.5,-0.5,-4.5),
    clusterMaxBin = cms.vdouble(19.5,383.5,5.5),
)
