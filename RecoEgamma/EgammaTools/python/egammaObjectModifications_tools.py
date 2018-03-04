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


    
def makeEnergyScaleAndSmearingSysModifier(eleProdName,phoProdName):
    energyScaleAndSmearing = cms.PSet(
        modifierName    = cms.string('EGExtraInfoModifierFromFloatValueMaps'),
        electron_config = cms.PSet(
            energyScaleUp = cms.InputTag(eleProdName,"EGMscaleUpUncertainty"),
            energyScaleDown = cms.InputTag(eleProdName,"EGMscaleDownUncertainty"),
            energyScaleStatUp = cms.InputTag(eleProdName,"EGMscaleStatUpUncertainty"),
            energyScaleStatDown = cms.InputTag(eleProdName,"EGMscaleStatDownUncertainty"),
            energyScaleSystUp = cms.InputTag(eleProdName,"EGMscaleSystUpUncertainty"),
            energyScaleSystDown = cms.InputTag(eleProdName,"EGMscaleSystDownUncertainty"),
            energyScaleGainUp = cms.InputTag(eleProdName,"EGMscaleGainUpUncertainty"),
            energyScaleGainDown = cms.InputTag(eleProdName,"EGMscaleGainDownUncertainty"),
            energySmearUp = cms.InputTag(eleProdName,"EGMresolutionUpUncertainty"),
            energySmearDown = cms.InputTag(eleProdName,"EGMresolutionDownUncertainty"),
            energySmearRhoUp = cms.InputTag(eleProdName,"EGMresolutionRhoUpUncertainty"),
            energySmearRhoDown = cms.InputTag(eleProdName,"EGMresolutionRhoDownUncertainty"),
            energySmearPhiUp = cms.InputTag(eleProdName,"EGMresolutionPhiUpUncertainty"),
            energySmearPhiDown = cms.InputTag(eleProdName,"EGMresolutionPhiDownUncertainty"),
            energyScaleValue = cms.InputTag(eleProdName,"EGMscale"),
            energySmearValue = cms.InputTag(eleProdName,"EGMsmear"),
            energySmearNrSigma = cms.InputTag(eleProdName,"EGMsmearNrSigma"),
            energyEcalPreCorr = cms.InputTag(eleProdName,"EGMecalEnergyPreCorr"),
            energyEcalErrPreCorr = cms.InputTag(eleProdName,"EGMecalEnergyErrPreCorr"),
            energyEcalTrkPreCorr = cms.InputTag(eleProdName,"EGMecalTrkEnergyPreCorr"),
            energyEcalTrkErrPreCorr = cms.InputTag(eleProdName,"EGMecalTrkEnergyErrPreCorr" ),
            energyEcalTrkPostCorr = cms.InputTag(eleProdName,"EGMecalTrkEnergy"),
            energyEcalTrkErrPostCorr = cms.InputTag(eleProdName,"EGMecalTrkEnergyErr"),
            ),
        photon_config   = cms.PSet( 
            energyScaleUp = cms.InputTag(phoProdName,"EGMscaleUpUncertainty"),
            energyScaleDown = cms.InputTag(phoProdName,"EGMscaleDownUncertainty"),
            energyScaleStatUp = cms.InputTag(phoProdName,"EGMscaleStatUpUncertainty"),
            energyScaleStatDown = cms.InputTag(phoProdName,"EGMscaleStatDownUncertainty"),
            energyScaleSystUp = cms.InputTag(phoProdName,"EGMscaleSystUpUncertainty"),
            energyScaleSystDown = cms.InputTag(phoProdName,"EGMscaleSystDownUncertainty"),
            energyScaleGainUp = cms.InputTag(phoProdName,"EGMscaleGainUpUncertainty"),
            energyScaleGainDown = cms.InputTag(phoProdName,"EGMscaleGainDownUncertainty"),
            energySmearUp = cms.InputTag(phoProdName,"EGMresolutionUpUncertainty"),
            energySmearDown = cms.InputTag(phoProdName,"EGMresolutionDownUncertainty"),
            energySmearRhoUp = cms.InputTag(phoProdName,"EGMresolutionRhoUpUncertainty"),
            energySmearRhoDown = cms.InputTag(phoProdName,"EGMresolutionRhoDownUncertainty"),
            energySmearPhiUp = cms.InputTag(phoProdName,"EGMresolutionPhiUpUncertainty"),
            energySmearPhiDown = cms.InputTag(phoProdName,"EGMresolutionPhiDownUncertainty"),
            energyScaleValue = cms.InputTag(phoProdName,"EGMscale"),
            energySmearValue = cms.InputTag(phoProdName,"EGMsmear"),
            energySmearNrSigma = cms.InputTag(phoProdName,"EGMsmearNrSigma"),
            energyEcalPreCorr = cms.InputTag(phoProdName,"EGMecalEnergyPreCorr"),
            energyEcalErrPreCorr = cms.InputTag(phoProdName,"EGMecalEnergyErrPreCorr"),
            )
        )
    return energyScaleAndSmearing
    
from RecoEgamma.EgammaTools.egammaObjectModificationsInMiniAOD_cff import reducedEgammaEnergyScaleAndSmearingModifier                                   
def appendReducedEgammaEnergyScaleAndSmearingModifier(modifiers):
    modifiers.append(reducedEgammaEnergyScaleAndSmearingModifier)
