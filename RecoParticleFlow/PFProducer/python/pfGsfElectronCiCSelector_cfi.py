import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.cutsInCategoriesElectronIdentificationV06_cfi import *

electronsWithPresel = cms.EDFilter("GsfElectronSelector",
                                   src = cms.InputTag("ecalDrivenGsfElectrons"),
                                   cut = cms.string("pt > 10 && ecalDrivenSeed && passingCutBasedPreselection"),
                                   )

electronsCiCLoose = cms.EDFilter("EleIdCutBased",
                                 src = cms.InputTag("electronsWithPresel"),
                                 algorithm = cms.string("eIDCB"),
                                 threshold = cms.double(14.5),
                                 electronIDType = eidLooseMC.electronIDType,
                                 electronQuality = eidLooseMC.electronQuality,
                                 electronVersion = eidLooseMC.electronVersion,
                                 additionalCategories = eidLooseMC.additionalCategories,
                                 classbasedlooseEleIDCutsV06 = eidLooseMC.classbasedlooseEleIDCutsV06,
                                 etBinning = cms.bool(False),
                                 version = cms.string(""),
                                 verticesCollection = cms.InputTag('offlinePrimaryVertices'),
                                 reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                                 reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                                 )
