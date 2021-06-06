#
# While reducing the AOD data tier produced in 80X to miniAOD, the electron
# MVAs are computed before slimming. However, the object updators in
# egammaObjectModificationsInMiniAOD_cff are written for pat::Electrons.
# Therefore, we must adapt the object modifiers to the AOD level such that the
# MVA producer can run on electrons that are updated and correctly store the
# conversion rejection variables.
#
# Little annoyance: updating the object also requires computing the HEEP value
# maps for the gedGsfElectrons, even though they are recomputed later from the
# reducedEgamma collections. Unfortunately we can't use these HEEP value maps
# that already exists, because reducedEgamma in turn depends on the
# electronMVAValueMap producer. Hence, this is a problem of circular
# dependency.
#

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from RecoEgamma.EgammaTools.egammaObjectModificationsInMiniAOD_cff import (
    egamma8XObjectUpdateModifier,
    egamma9X105XUpdateModifier,
)
from RecoEgamma.ElectronIdentification.heepIdVarValueMapProducer_cfi import heepIDVarValueMaps

heepIDVarValueMapsAOD = heepIDVarValueMaps.copy()
heepIDVarValueMapsAOD.dataFormat = 1

gsfElectron8XObjectUpdateModifier = egamma8XObjectUpdateModifier.clone(
    ecalRecHitsEB="reducedEcalRecHitsEB", ecalRecHitsEE="reducedEcalRecHitsEE"
)
gsfElectron9X105XUpdateModifier = egamma9X105XUpdateModifier.clone(
    eleCollVMsAreKeyedTo="gedGsfElectrons",
    eleTrkIso="heepIDVarValueMapsAOD:eleTrkPtIso",
    eleTrkIso04="heepIDVarValueMapsAOD:eleTrkPtIso04",
    conversions="allConversions",
    ecalRecHitsEB="reducedEcalRecHitsEB",
    ecalRecHitsEE="reducedEcalRecHitsEE",
)

# we have dataformat changes to 106X so to read older releases we use egamma updators
gedGsfElectronsFrom80XTo106X = cms.EDProducer(
    "ModifiedGsfElectronProducer",
    src=cms.InputTag("gedGsfElectrons"),
    modifierConfig=cms.PSet(
        modifications=cms.VPSet(gsfElectron8XObjectUpdateModifier, gsfElectron9X105XUpdateModifier)
    ),
)

gedGsfElectronsFrom80XTo106XTask = cms.Task(heepIDVarValueMapsAOD, gedGsfElectronsFrom80XTo106X)

gedGsfElectronsFrom94XTo106X = cms.EDProducer(
    "ModifiedGsfElectronProducer",
    src=cms.InputTag("gedGsfElectrons"),
    modifierConfig=cms.PSet(
        modifications=cms.VPSet(gsfElectron9X105XUpdateModifier)
    ),
)

gedGsfElectronsFrom94XTo106XTask = cms.Task(heepIDVarValueMapsAOD, gedGsfElectronsFrom94XTo106X)
