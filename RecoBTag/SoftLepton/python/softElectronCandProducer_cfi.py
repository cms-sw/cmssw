import FWCore.ParameterSet.Config as cms
from RecoBTag.SoftLepton.softPFElectronCleaner_cfi import *

softElectronCands = cms.EDProducer("SoftElectronCandProducer",
  looseSoftPFElectronCleanerBarrelCuts,
  looseSoftPFElectronCleanerForwardCuts,
  electrons = cms.InputTag("gsfElectrons")
)
