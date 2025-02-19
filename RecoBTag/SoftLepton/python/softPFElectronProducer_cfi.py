import FWCore.ParameterSet.Config as cms
from RecoBTag.SoftLepton.softPFElectronCleaner_cfi import *

softPFElectrons = cms.EDProducer("SoftPFElectronProducer",
  looseSoftPFElectronCleanerBarrelCuts,
  looseSoftPFElectronCleanerForwardCuts,
  Electrons = cms.InputTag("gsfElectrons")
)
