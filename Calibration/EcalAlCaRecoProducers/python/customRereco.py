import FWCore.ParameterSet.Config as cms

def EcalRecal(process):
    recalibElectronSrc = cms.InputTag("electronRecalibSCAssociator")
    process.alCaIsolatedElectrons.srcLabels = cms.VInputTag( "electronRecalibSCAssociator")
    process.alCaIsolatedElectrons.electronLabel = "electronRecalibSCAssociator"
    process.alCaIsolatedElectrons.ebRecHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
    process.alCaIsolatedElectrons.eeRecHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
    process.alCaIsolatedElectrons.EESuperClusterCollection = process.reducedEcalRecHitsES.EndcapSuperClusterCollection

    process.selectedElectrons.src = cms.InputTag("electronRecalibSCAssociator")
    process.PassingVetoId.src = recalibElectronSrc
#    process.myEleCollection = cms.InputTag('electronRecalibSCAssociator')
    return process
