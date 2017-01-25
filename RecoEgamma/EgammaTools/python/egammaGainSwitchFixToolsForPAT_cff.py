import FWCore.ParameterSet.Config as cms 

def customizeGSFixForPAT(process): 

    process.reducedEgamma.gsfElectrons = cms.InputTag("gedGsfElectronsGSFixed")
    process.reducedEgamma.gsfElectronsPFValMap = cms.InputTag("particleBasedIsolationGSFixed","gedGsfElectrons")
    process.reducedEgamma.gsfElectronPFClusterIsoSources = cms.VInputTag(
        cms.InputTag("electronEcalPFClusterIsolationProducerGSFixed"),
        cms.InputTag("electronHcalPFClusterIsolationProducerGSFixed"),
        )
    process.reducedEgamma.gsfElectronIDSources = cms.VInputTag(
        cms.InputTag("eidLooseGSFixed"),
        cms.InputTag("eidRobustHighEnergyGSFixed"),
        cms.InputTag("eidRobustLooseGSFixed"),
        cms.InputTag("eidRobustTightGSFixed"),
        cms.InputTag("eidTightGSFixed"),
        )
    process.reducedEgamma.photons = cms.InputTag("gedPhotonsGSFixed")
    process.reducedEgamma.conversions = cms.InputTag("allConversionsGSFixed")
    process.reducedEgamma.singleConversions = cms.InputTag("particleFlowEGammaGSFixed")
    process.reducedEgamma.photonsPFValMap = cms.InputTag("particleBasedIsolationGSFixed","gedPhotons")
    process.reducedEgamma.photonPFClusterIsoSources = cms.VInputTag(
        cms.InputTag("photonEcalPFClusterIsolationProducerGSFixed"),
        cms.InputTag("photonHcalPFClusterIsolationProducerGSFixed"),
        )
    process.reducedEgamma.photonIDSources = cms.VInputTag(
        cms.InputTag("PhotonCutBasedIDLooseGSFixed"),
        cms.InputTag("PhotonCutBasedIDLooseEMGSFixed"),    
        cms.InputTag("PhotonCutBasedIDTightGSFixed")
        )
    process.reducedEgamma.barrelEcalHits = cms.InputTag("ecalMultiAndGSWeightRecHitEB")
    process.reducedEgamma.endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE")
    process.reducedEgamma.preshowerEcalHits = cms.InputTag("reducedEcalRecHitsES")

    return process
