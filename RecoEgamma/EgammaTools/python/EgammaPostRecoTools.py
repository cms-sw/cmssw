
import FWCore.ParameterSet.Config as cms



#define the default IDs to produce in VID
_defaultEleIDModules =  [ 'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff',
                        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V1_cff',
                        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V1_cff', 
                        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V1_cff',
                        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Summer16_80X_V1_cff',
                        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff',
                        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff',
                        ]
_defaultPhoIDModules =  [ 'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Fall17_94X_V1_TrueVtx_cff',
                        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1_cff', 
                        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1p1_cff', 
                        'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring16_V2p2_cff',
                        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring16_nonTrig_V1_cff'
                        ]


def _setupEgammaPostRECOSequence(process,applyEnergyCorrections=False,applyVIDOnCorrectedEgamma=False):
    if applyVIDOnCorrectedEgamma:
        raise RuntimeError('Error in postRecoEgammaTools, _setupEgammaPostRECOSequence can not currently apply VID on corrected E/gammas in AOD due to ValueMap issues'.format(applyEnergyCorrections,applyVIDOnCorrectedEgamma))
    if applyEnergyCorrections: 
        raise RuntimeError('Error in postRecoEgammaTools, _setupEgammaPostRECOSequence can not currently produce new collections with E/gamma energies corrected in AOD due to ValueMap issues'.format(applyEnergyCorrections,applyVIDOnCorrectedEgamma))

    phoSrc = cms.InputTag('gedPhotons')
    eleSrc = cms.InputTag('gedGsfElectrons')

    process.load('RecoEgamma.EgammaTools.calibratedEgammas_cff')
    #this code is just here waiting for VM issues to be solved, it cant be called right now
    if applyEnergyCorrections:
        phoCalibSrc = cms.InputTag('gedPhotons',processName=cms.InputTag.skipCurrentProcess())
        eleCalibSrc = cms.InputTag('gedGsfElectrons',processName=cms.InputTag.skipCurrentProcess())
        
        process.gedGsfElectrons = process.calibratedElectrons.clone(src=eleCalibSrc,
                                                                    produceCalibratedObjs=True)
        process.gedPhotons = process.calibratedPhotons.clone(src=phoCalibSrc,
                                                             produceCalibratedObjs=True)

        process.egammaScaleSmearTask = cms.Task(process.gedGsfElectrons,
                                                process.gedPhotons
                                                )
    else:
        phoCalibSrc = phoSrc
        eleCalibSrc = eleSrc
        process.calibratedElectrons.produceCalibratedObjs = False 
        process.calibratedPhotons.produceCalibratedObjs = False 
        process.calibratedElectrons.src = eleCalibSrc
        process.calibratedPhotons.src = phoCalibSrc

        process.egammaScaleSmearTask = cms.Task(process.calibratedElectrons,
                                                process.calibratedPhotons
                                                )
    process.egmGsfElectronIDs.physicsObjectSrc = eleSrc
    process.egmPhotonIDs.physicsObjectSrc = phoSrc
    process.electronMVAValueMapProducer.src = eleSrc
    process.photonMVAValueMapProducer.src = phoSrc
    process.photonIDValueMapProducer.src = phoSrc
    process.egmPhotonIsolation.srcToIsolate = phoSrc
    
    if hasattr(process,'heepIDVarValueMaps'):
        process.heepIDVarValueMaps.elesAOD = eleSrc
        
        
                                          
"""
This function loads the calibrated producers calibratedPatElectrons,calibratedPatPhotons, 
sets VID & other modules to the correct electron/photon source,
loads up the modifiers and which then creates a new slimmedElectrons,slimmedPhotons collection
with VID and scale and smearing all loaded in
"""

def _setupEgammaPostRECOSequenceMiniAOD(process,applyEnergyCorrections=False,applyVIDOnCorrectedEgamma=False):

    
    if applyEnergyCorrections != applyVIDOnCorrectedEgamma:
        raise RuntimeError('Error, applyEnergyCorrections {} and applyVIDOnCorrectedEgamma {} must be equal to each other for now,\n functionality for them to be different isnt yet availible'.format(applyEnergyCorrections,applyVIDOnCorrectedEgamma))


    phoSrc = cms.InputTag('slimmedPhotons',processName=cms.InputTag.skipCurrentProcess())
    eleSrc = cms.InputTag('slimmedElectrons',processName=cms.InputTag.skipCurrentProcess())
    phoCalibSrc = cms.InputTag('slimmedPhotons',processName=cms.InputTag.skipCurrentProcess())
    eleCalibSrc = cms.InputTag('slimmedElectrons',processName=cms.InputTag.skipCurrentProcess())

    process.load('RecoEgamma.EgammaTools.calibratedEgammas_cff')
    process.calibratedPatElectrons.src = eleCalibSrc
    process.calibratedPatPhotons.src = phoCalibSrc
    
    if applyEnergyCorrections and applyVIDOnCorrectedEgamma:
        phoSrc = cms.InputTag('calibratedPatPhotons')
        eleSrc = cms.InputTag('calibratedPatElectrons') 
        process.calibratedPatElectrons.produceCalibratedObjs = True
        process.calibratedPatPhotons.produceCalibratedObjs = True
    if not applyEnergyCorrections:
        process.calibratedPatElectrons.produceCalibratedObjs = False 
        process.calibratedPatPhotons.produceCalibratedObjs = False 
        
    process.egmGsfElectronIDs.physicsObjectSrc = eleSrc
    process.egmPhotonIDs.physicsObjectSrc = phoSrc
    process.electronMVAValueMapProducer.srcMiniAOD = eleSrc
    process.photonMVAValueMapProducer.srcMiniAOD = phoSrc
    process.photonIDValueMapProducer.srcMiniAOD = phoSrc
    process.egmPhotonIsolation.srcToIsolate = phoSrc
    if hasattr(process,'heepIDVarValueMaps'):
        process.heepIDVarValueMaps.elesMiniAOD = eleSrc



    from RecoEgamma.EgammaTools.egammaObjectModificationsInMiniAOD_cff import egamma_modifications
    from RecoEgamma.EgammaTools.egammaObjectModifications_tools import makeVIDBitsModifier,makeVIDinPATIDsModifier,makeEnergyScaleAndSmearingSysModifier                                     
    egamma_modifications.append(makeVIDBitsModifier(process,"egmGsfElectronIDs","egmPhotonIDs"))
    egamma_modifications.append(makeVIDinPATIDsModifier(process,"egmGsfElectronIDs","egmPhotonIDs"))
    egamma_modifications.append(makeEnergyScaleAndSmearingSysModifier("calibratedPatElectrons","calibratedPatPhotons"))

    #add the HEEP trk isol to the slimmed electron
    for pset in egamma_modifications:
            if pset.hasParameter("modifierName") and pset.modifierName == cms.string('EGExtraInfoModifierFromFloatValueMaps'):
                pset.electron_config.heepTrkPtIso = cms.InputTag("heepIDVarValueMaps","eleTrkPtIso")
                break

    for pset in egamma_modifications:
        pset.overrideExistingValues = cms.bool(True)
        if hasattr(pset,"electron_config"): pset.electron_config.electronSrc = eleSrc
        if hasattr(pset,"photon_config"): pset.photon_config.photonSrc = phoSrc

    process.slimmedElectrons = cms.EDProducer("ModifiedElectronProducer",
                                              src=eleSrc,
                                              modifierConfig = cms.PSet(
                                                  modifications = egamma_modifications
                                                  )
                                              )
    process.slimmedPhotons = cms.EDProducer("ModifiedPhotonProducer",
                                            src=phoSrc,
                                            modifierConfig = cms.PSet(
                                                modifications = egamma_modifications
                                                )
                                            )
    process.egammaScaleSmearTask = cms.Task(process.calibratedPatElectrons,
                                            process.slimmedElectrons,
                                            process.calibratedPatPhotons,
                                            process.slimmedPhotons
                                            )



def setupEgammaPostRecoSeq(process,
                           applyEnergyCorrections=False,
                           applyVIDOnCorrectedEgamma=False,
                           isMiniAOD=True,
                           eleIDModules=_defaultEleIDModules,
                           phoIDModules=_defaultPhoIDModules):

    from PhysicsTools.SelectorUtils.tools.vid_id_tools import switchOnVIDElectronIdProducer,switchOnVIDPhotonIdProducer,setupAllVIDIdsInModule,DataFormat,setupVIDElectronSelection,setupVIDPhotonSelection
    # turn on VID producer, indicate data format  to be
    # DataFormat.AOD or DataFormat.MiniAOD, as appropriate
    if isMiniAOD:
        switchOnVIDElectronIdProducer(process,DataFormat.MiniAOD)
        switchOnVIDPhotonIdProducer(process,DataFormat.MiniAOD)
    else:
        switchOnVIDElectronIdProducer(process,DataFormat.AOD)
        switchOnVIDPhotonIdProducer(process,DataFormat.AOD)


    for idmod in eleIDModules:
        setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection)
    for idmod in phoIDModules:
        setupAllVIDIdsInModule(process,idmod,setupVIDPhotonSelection)
    
    if isMiniAOD:
        _setupEgammaPostRECOSequenceMiniAOD(process,applyEnergyCorrections,applyVIDOnCorrectedEgamma)
    else:
        _setupEgammaPostRECOSequence(process,applyEnergyCorrections,applyVIDOnCorrectedEgamma)
    
    process.egammaScaleSmearSeq = cms.Sequence( process.egammaScaleSmearTask)
    process.egammaPostRecoSeq   = cms.Sequence( process.egammaScaleSmearSeq*
                                                process.egmGsfElectronIDSequence*
                                                process.egmPhotonIDSequence )
    return process
