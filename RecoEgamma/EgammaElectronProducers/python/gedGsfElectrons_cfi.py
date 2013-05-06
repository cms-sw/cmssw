import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi import *

gedGsfElectrons=ecalDrivenGsfElectrons.clone()
gedGsfElectrons.gsfElectronCoresTag="gedGsfElectronCores"
