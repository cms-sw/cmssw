import FWCore.ParameterSet.Config as cms

collectionName = 'calibratedPatElectrons'
configElectronEnergyShifter = cms.PSet(


    calibratedElectrons = cms.InputTag(collectionName),

    scaleStatUpUncertainty = cms.InputTag('%s:EGMscaleStatUpUncertainty' % collectionName),
    scaleStatDownUncertainty = cms.InputTag('%s:EGMscaleStatDownUncertainty' % collectionName),
    scaleSystUpUncertainty = cms.InputTag('%s:EGMscaleSystUpUncertainty' % collectionName),
    scaleSystDownUncertainty = cms.InputTag('%s:EGMscaleSystDownUncertainty' % collectionName),
    scaleGainUpUncertainty = cms.InputTag('%s:EGMscaleGainUpUncertainty' % collectionName),
    scaleGainDownUncertainty = cms.InputTag('%s:EGMscaleGainDownUncertainty' % collectionName),
    resolutionRhoUpUncertainty = cms.InputTag('%s:EGMresolutionRhoUpUncertainty' % collectionName),
    resolutionRhoDownUncertainty = cms.InputTag('%s:EGMresolutionRhoDownUncertainty' % collectionName),
    resolutionPhiUpUncertainty = cms.InputTag('%s:EGMresolutionPhiUpUncertainty' % collectionName),
    resolutionPhiDownUncertainty = cms.InputTag('%s:EGMresolutionPhiDownUncertainty' % collectionName),
    
    scaleUpUncertainty = cms.InputTag('%s:EGMscaleUpUncertainty' % collectionName),
    scaleDownUncertainty = cms.InputTag('%s:EGMscaleDownUncertainty' % collectionName),
    resolutionUpUncertainty = cms.InputTag('%s:EGMresolutionUpUncertainty' % collectionName),
    resolutionDownUncertainty = cms.InputTag('%s:EGMresolutionDownUncertainty' % collectionName)
    
    )

collectionName = 'calibratedPatPhotons'
configPhotonEnergyShifter = cms.PSet(
    
    calibratedPhotons = cms.InputTag(collectionName),

    scaleStatUpUncertainty = cms.InputTag('%s:EGMscaleStatUpUncertainty' % collectionName),
    scaleStatDownUncertainty = cms.InputTag('%s:EGMscaleStatDownUncertainty' % collectionName),
    scaleSystUpUncertainty = cms.InputTag('%s:EGMscaleSystUpUncertainty' % collectionName),
    scaleSystDownUncertainty = cms.InputTag('%s:EGMscaleSystDownUncertainty' % collectionName),
    scaleGainUpUncertainty = cms.InputTag('%s:EGMscaleGainUpUncertainty' % collectionName),
    scaleGainDownUncertainty = cms.InputTag('%s:EGMscaleGainDownUncertainty' % collectionName),
    resolutionRhoUpUncertainty = cms.InputTag('%s:EGMresolutionRhoUpUncertainty' % collectionName),
    resolutionRhoDownUncertainty = cms.InputTag('%s:EGMresolutionRhoDownUncertainty' % collectionName),
    resolutionPhiUpUncertainty = cms.InputTag('%s:EGMresolutionPhiUpUncertainty' % collectionName),
    resolutionPhiDownUncertainty = cms.InputTag('%s:EGMresolutionPhiDownUncertainty' % collectionName),
    
    scaleUpUncertainty = cms.InputTag('%s:EGMscaleUpUncertainty' % collectionName),
    scaleDownUncertainty = cms.InputTag('%s:EGMscaleDownUncertainty' % collectionName),
    resolutionUpUncertainty = cms.InputTag('%s:EGMresolutionUpUncertainty' % collectionName),
    resolutionDownUncertainty = cms.InputTag('%s:EGMresolutionDownUncertainty' % collectionName)
    
    )
