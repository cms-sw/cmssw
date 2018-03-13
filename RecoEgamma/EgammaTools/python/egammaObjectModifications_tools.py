import FWCore.ParameterSet.Config as cms

"""
makes a modifier config to load all the cutbased VID bits into the pat::Electron/pat::Photon userdata
any ID which starts with "mva" will not be stored
"""
def makeVIDBitsModifier(process,eleVIDModuleName,phoVIDModuleName):
    vidCutBitsModifier = cms.PSet(
        modifierName    = cms.string('EGExtraInfoModifierFromUIntToIntValueMaps'),
        electron_config = cms.PSet(),
        photon_config = cms.PSet()
        )
    
    phoVIDModule = getattr(process,phoVIDModuleName)
    for egid in phoVIDModule.physicsObjectIDs:
        if egid.idDefinition.idName.value().find("mva")!=0:
            setattr(vidCutBitsModifier.photon_config,egid.idDefinition.idName.value(),cms.InputTag(phoVIDModuleName+':'+egid.idDefinition.idName.value()+"Bitmap"))

    eleVIDModule = getattr(process,eleVIDModuleName)
    for egid in eleVIDModule.physicsObjectIDs:  
        if egid.idDefinition.idName.value().find("mva")!=0:
            setattr(vidCutBitsModifier.electron_config,egid.idDefinition.idName.value(),cms.InputTag(eleVIDModuleName+':'+egid.idDefinition.idName.value()+"Bitmap"))

    return vidCutBitsModifier

"""
make a modifer config to load all the VID ids into the pat::Electron/pat::Photon electron/photonIDs
"""
def makeVIDinPATIDsModifier(process,eleVIDModuleName,phoVIDModuleName):
    vidInPATIDsModifier = cms.PSet(
        modifierName    = cms.string('EGExtraInfoModifierFromEGIDValueMaps'),
        electron_config = cms.PSet(),
        photon_config = cms.PSet()
        )
    phoVIDModule = getattr(process,phoVIDModuleName)
    for egid in phoVIDModule.physicsObjectIDs:      
        setattr(vidInPATIDsModifier.photon_config,egid.idDefinition.idName.value(),cms.InputTag(phoVIDModuleName+':'+egid.idDefinition.idName.value()))

    eleVIDModule = getattr(process,eleVIDModuleName)
    for egid in eleVIDModule.physicsObjectIDs:
        setattr(vidInPATIDsModifier.electron_config,egid.idDefinition.idName.value(),cms.InputTag(eleVIDModuleName+':'+egid.idDefinition.idName.value()))
    
    return vidInPATIDsModifier

"""
make a modifer config to load all scale&smearing info into the pat::Electron/pat::Photon 
takes the names of the electron and photon producer modules
"""
def makeEnergyScaleAndSmearingSysModifier(eleProdName,phoProdName):
    energyScaleAndSmearing = cms.PSet(
        modifierName    = cms.string('EGExtraInfoModifierFromFloatValueMaps'),
        electron_config = cms.PSet(),
        photon_config = cms.PSet()
        )

    from RecoEgamma.EgammaTools.calibratedEgammas_cff import prefixName
    import RecoEgamma.EgammaTools.calibratedElectronProducer_cfi
    for valueMapName in RecoEgamma.EgammaTools.calibratedElectronProducer_cfi.calibratedElectronProducer.valueMapsStored:
        setattr(energyScaleAndSmearingModifier.electron_config,valueMapName,cms.InputTag(eleProdName,valueMapName))
    import RecoEgamma.EgammaTools.calibratedPhotonProducer_cfi
    for valueMapName in RecoEgamma.EgammaTools.calibratedPhotonProducer_cfi.calibratedPhotonProducer.valueMapsStored:
        setattr(reducedEgammaEnergyScaleAndSmearingModifier.electron_config,valueMapName,cms.InputTag(phoProdName,valueMapName))
    return energyScaleAndSmearing

"""
setups up the calibrated egamma producers and then adds a modifier which
will embed the scale & smearing info into the pat::Electron/Photons
"""
def _storeCalibratedEGEnergiesForRun2MiniAOD9XFall17(process):
    process.load("RecoEgamma.EgammaTools.calibratedEgammas_cfi")
    process.calibratedPhotons.produceCalibratedObjs = cms.bool(False)
    process.calibratedElectrons.produceCalibratedObjs = cms.bool(False)
    process.calibratedElectrons.src = process.patElectrons.electronSource
    process.calibratedPhotons.src = process.patPhotons.photonSource
    process.calibratedElectrons.recHitCollectionEB = cms.InputTag("reducedEgamma","reducedEBRecHits")
    process.calibratedElectrons.recHitCollectionEE = cms.InputTag("reducedEgamma","reducedEERecHits")
    process.calibratedPhotons.recHitCollectionEB = cms.InputTag("reducedEgamma","reducedEBRecHits")
    process.calibratedPhotons.recHitCollectionEE = cms.InputTag("reducedEgamma","reducedEERecHits")
    process.slimmingTask.add(process.calibratedPhotons)
    process.slimmingTask.add(process.calibratedElectrons)
    from RecoEgamma.EgammaTools.egammaObjectModificationsInMiniAOD_cff import egamma_modifications
    egamma_modifications.append(makeEnergyScaleAndSmearingSysModifier("calibratedElectrons","calibratedPhotons"))

from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
modifyEgammaObjectModifications_toolsForRun2MiniAOD9XFall17_ = run2_miniAOD_94XFall17.makeProcessModifier(_storeCalibratedEGEnergiesForRun2MiniAOD9XFall17)

