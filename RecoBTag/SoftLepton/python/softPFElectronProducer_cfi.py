import FWCore.ParameterSet.Config as cms
from RecoBTag.SoftLepton.softPFElectronCleaner_cfi import *

softPFElectrons = cms.EDProducer("SoftPFElectronProducer",
  looseSoftPFElectronCleanerBarrelCuts,
  looseSoftPFElectronCleanerForwardCuts,
  PFElectrons = cms.InputTag("particleFlow", "electrons")
)
