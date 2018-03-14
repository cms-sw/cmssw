import FWCore.ParameterSet.Config as cms

def calibrateReducedEgamma(process):
    process.load("RecoEgamma.EgammaTools.calibratedEgammas_cff")
    process.calibratedPhotons.produceCalibratedObjs = False
    process.calibratedElectrons.produceCalibratedObjs = False
    process.slimmingTask.add(process.calibratedPhotons)
    process.slimmingTask.add(process.calibratedElectrons)
    process.reducedEgamma.applyPhotonCalibOnData = False
    process.reducedEgamma.applyPhotonCalibOnMC = False
    process.reducedEgamma.applyGsfElectronCalibOnData = False
    process.reducedEgamma.applyGsfElectronCalibOnMC = False
    process.reducedEgamma.photonCalibEnergySource = cms.InputTag("calibratedPhotons","ecalEnergyPostCorr")
    process.reducedEgamma.photonCalibEnergyErrSource = cms.InputTag("calibratedPhotons","ecalEnergyErrPostCorr")
    process.reducedEgamma.gsfElectronCalibEnergySource = cms.InputTag("calibratedElectrons","ecalTrkEnergyPostCorr")
    process.reducedEgamma.gsfElectronCalibEnergyErrSource = cms.InputTag("calibratedElectrons","ecalTrkEnergyErrPostCorr")

    process.reducedEgamma.gsfElectronCalibEcalEnergySource = cms.InputTag("calibratedElectrons","ecalEnergyPostCorr")
    process.reducedEgamma.gsfElectronCalibEcalEnergyErrSource = cms.InputTag("calibratedElectrons","ecalEnergyErrPostCorr")
    from RecoEgamma.EgammaTools.calibratedEgammas_cff import prefixName
    import RecoEgamma.EgammaTools.calibratedElectronProducerTRecoGsfElectron_cfi
    for valueMapName in RecoEgamma.EgammaTools.calibratedElectronProducerTRecoGsfElectron_cfi.calibratedElectronProducerTRecoGsfElectron.valueMapsStored:
        process.reducedEgamma.gsfElectronFloatValueMapSources.append(cms.InputTag("calibratedElectrons",valueMapName))
        process.reducedEgamma.gsfElectronFloatValueMapOutput.append(prefixName("calibEle",valueMapName))
    import RecoEgamma.EgammaTools.calibratedPhotonProducerTRecoPhoton_cfi
    for valueMapName in RecoEgamma.EgammaTools.calibratedPhotonProducerTRecoPhoton_cfi.calibratedPhotonProducerTRecoPhoton.valueMapsStored:
        process.reducedEgamma.photonFloatValueMapSources.append(cms.InputTag("calibratedPhotons",valueMapName))
        process.reducedEgamma.photonFloatValueMapOutput.append(prefixName("calibPho",valueMapName))

