import FWCore.ParameterSet.Config as cms

# Whatever has been available in 758; already updates in 76X

electronVetoID50nsV1   = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-50ns-V1-standalone-veto")
electronLooseID50nsV1  = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-50ns-V1-standalone-loose")
electronMediumID50nsV1 = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-50ns-V1-standalone-medium")
electronTightID50nsV1  = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-50ns-V1-standalone-tight")

electronVetoID25nsV1   = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-veto")
electronLooseID25nsV1  = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-loose")
electronMediumID25nsV1 = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-medium")
electronTightID25nsV1  = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-tight")


# Effective areas for computing PU correction for isolations
effAreasConfigFile50ns = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Spring15/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_50ns.txt")

effAreasConfigFile25ns = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Spring15/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_25ns.txt")
