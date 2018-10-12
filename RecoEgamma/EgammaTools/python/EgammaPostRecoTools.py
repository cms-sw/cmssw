
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

#the new Fall17V2 modules are loaded as default if they exist in the release
#we do it this way as we can use the same script for all releases and people who
#dont want V2 can still use this script
_fall17V2PhoIDModules = [
    'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V2_cff'
    ]
_fall17V2EleIDModules = [
    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V2_cff',
    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff',
    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff'
    ]

import pkgutil
if pkgutil.find_loader(_fall17V2EleIDModules[0]) != None:
    _defaultEleIDModules.extend(_fall17V2EleIDModules)
else:
    print "EgammaPostRecoTools: Fall17V2 electron modules not found, running ID without them. If you want Fall17V2 IDs, please merge the approprate PR"

if pkgutil.find_loader(_fall17V2PhoIDModules[0]) != None:
    _defaultPhoIDModules.extend(_fall17V2PhoIDModules)
else:
    print "EgammaPostRecoTools: Fall17V2 photons modules not found, running ID without them. If you want Fall17V2 IDs, please merge the approprate PR"

def _getEnergyCorrectionFile(era):
    if era=="2017-Nov17ReReco":
        return "EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2017_17Nov2017_v1_ele_unc"
    if era=="2016-Legacy":
        return "EgammaAnalysis/ElectronTools/data/ScalesSmearings/Legacy2016_07Aug2017_FineEtaR9_v3_ele_unc"
    if era=="2016-Feb17ReMiniAOD":
        raise RuntimeError('Error in postRecoEgammaTools, era 2016-Feb17ReMiniAOD is not currently implimented') 
    raise RuntimeError('Error in postRecoEgammaTools, era '+era+' not recognised. Allowed eras are 2017-Nov17ReReco, 2016-Legacy, 2016-Feb17ReMiniAOD')

def _is80XRelease(era):
    if era=="2016-Legacy" or era=="2016-Feb17ReMiniAOD": return True
    elif era!="2017-Nov17ReReco":
        raise RuntimeError('Error in postRecoEgammaTools, era '+era+' not recognised. Allowed eras are 2017-Nov17ReReco, 2016-Legacy, 2016-Feb17ReMiniAOD')


def _setupEgammaPostRECOSequence(process,applyEnergyCorrections=False,applyVIDOnCorrectedEgamma=False,era="2017-Nov17ReReco",runVID=True,applyEPCombBug=False):
    if applyVIDOnCorrectedEgamma:
        raise RuntimeError('Error in postRecoEgammaTools, _setupEgammaPostRECOSequence can not currently apply VID on corrected E/gammas in AOD due to ValueMap issues'.format(applyEnergyCorrections,applyVIDOnCorrectedEgamma))
    if applyEnergyCorrections: 
        raise RuntimeError('Error in postRecoEgammaTools, _setupEgammaPostRECOSequence can not currently produce new collections with E/gamma energies corrected in AOD due to ValueMap issues'.format(applyEnergyCorrections,applyVIDOnCorrectedEgamma))

    phoSrc = cms.InputTag('gedPhotons')
    eleSrc = cms.InputTag('gedGsfElectrons')

    if _is80XRelease(era): 
        print "EgammaPostRecoTools: begin warning:"
        print "   when running in 80X AOD, currenly do not fill 94X new data members "
        print "   members not filled: "
        print "      eles: e2x5Left, e2x5Right, e2x5Top, e2x5Bottom, nSaturatedXtals, isSeedSaturated"
        print "      phos: nStaturatedXtals, isSeedSaturated"
        print "   these are needed for the 80X energy regression if you are running it (if you dont know if  you are, you are not)"
        print "   the miniAOD method fills them correctly"
        print "   if you have a use case for AOD and need those members, contact e/gamma pog and we can find a solution"
        print "EgammaPostRecoTools: end warning"

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
    if runVID:
        process.egmGsfElectronIDs.physicsObjectSrc = eleSrc
        process.egmPhotonIDs.physicsObjectSrc = phoSrc
        process.electronMVAValueMapProducer.src = eleSrc
        process.photonMVAValueMapProducer.src = phoSrc
        process.photonIDValueMapProducer.src = phoSrc
        process.egmPhotonIsolation.srcToIsolate = phoSrc

    energyCorrectionFile = _getEnergyCorrectionFile(era)
    process.calibratedElectrons.correctionFile = energyCorrectionFile
    process.calibratedPhotons.correctionFile = energyCorrectionFile

    if applyEPCombBug:
        process.calibratedElectrons.useSmearCorrEcalEnergyErrInComb=True
    else:
        process.calibratedElectrons.useSmearCorrEcalEnergyErrInComb=False

    
    if runVID and hasattr(process,'heepIDVarValueMaps'):
        process.heepIDVarValueMaps.elesAOD = eleSrc
        process.heepIDVarValueMaps.dataFormat = 1
    if hasattr(process,'packedCandsForTkIso') and era.find("2016")!=-1:
        process.packedCandsForTkIso.chargedHadronIsolation = ""
        
                                          
"""
This function loads the calibrated producers calibratedPatElectrons,calibratedPatPhotons, 
sets VID & other modules to the correct electron/photon source,
loads up the modifiers and which then creates a new slimmedElectrons,slimmedPhotons collection
with VID and scale and smearing all loaded in
"""

def _setupEgammaPostRECOSequenceMiniAOD(process,applyEnergyCorrections=False,applyVIDOnCorrectedEgamma=False,era="2017-Nov17ReReco",runVID=True,applyEPCombBug=False):

    
    if applyEnergyCorrections != applyVIDOnCorrectedEgamma:
        raise RuntimeError('Error, applyEnergyCorrections {} and applyVIDOnCorrectedEgamma {} must be equal to each other for now,\n functionality for them to be different isnt yet availible'.format(applyEnergyCorrections,applyVIDOnCorrectedEgamma))


    phoSrc = cms.InputTag('slimmedPhotons',processName=cms.InputTag.skipCurrentProcess())
    eleSrc = cms.InputTag('slimmedElectrons',processName=cms.InputTag.skipCurrentProcess())
    phoCalibSrc = cms.InputTag('slimmedPhotons',processName=cms.InputTag.skipCurrentProcess())
    eleCalibSrc = cms.InputTag('slimmedElectrons',processName=cms.InputTag.skipCurrentProcess())

    process.load('RecoEgamma.EgammaTools.calibratedEgammas_cff')
    process.calibratedPatElectrons.src = eleCalibSrc
    process.calibratedPatPhotons.src = phoCalibSrc
    
    energyCorrectionFile = _getEnergyCorrectionFile(era)
    process.calibratedPatElectrons.correctionFile = energyCorrectionFile
    process.calibratedPatPhotons.correctionFile = energyCorrectionFile
    if applyEPCombBug:
        process.calibratedPatElectrons.useSmearCorrEcalEnergyErrInComb=True
    else:
        process.calibratedPatElectrons.useSmearCorrEcalEnergyErrInComb=False

    if applyEnergyCorrections and applyVIDOnCorrectedEgamma:
        phoSrc = cms.InputTag('calibratedPatPhotons')
        eleSrc = cms.InputTag('calibratedPatElectrons') 
        process.calibratedPatElectrons.produceCalibratedObjs = True
        process.calibratedPatPhotons.produceCalibratedObjs = True
    if not applyEnergyCorrections:
        process.calibratedPatElectrons.produceCalibratedObjs = False 
        process.calibratedPatPhotons.produceCalibratedObjs = False 

    if runVID:
        process.egmGsfElectronIDs.physicsObjectSrc = eleSrc
        process.egmPhotonIDs.physicsObjectSrc = phoSrc
        process.electronMVAValueMapProducer.srcMiniAOD = eleSrc
        process.photonMVAValueMapProducer.srcMiniAOD = phoSrc
        process.photonIDValueMapProducer.srcMiniAOD = phoSrc
        process.egmPhotonIsolation.srcToIsolate = phoSrc

    if runVID and hasattr(process,'heepIDVarValueMaps'):
        process.heepIDVarValueMaps.elesMiniAOD = eleSrc
        process.heepIDVarValueMaps.dataFormat = 2


    from RecoEgamma.EgammaTools.egammaObjectModificationsInMiniAOD_cff import egamma_modifications,egamma8XLegacyEtScaleSysModifier,egamma8XObjectUpdateModifier
    from RecoEgamma.EgammaTools.egammaObjectModifications_tools import makeVIDBitsModifier,makeVIDinPATIDsModifier,makeEnergyScaleAndSmearingSysModifier  
    if _is80XRelease(era): egamma_modifications.append(egamma8XObjectUpdateModifier) #if we were generated in 80X, we need fill in missing data members in 94X
    if runVID:
        egamma_modifications.append(makeVIDBitsModifier(process,"egmGsfElectronIDs","egmPhotonIDs"))
        egamma_modifications.append(makeVIDinPATIDsModifier(process,"egmGsfElectronIDs","egmPhotonIDs"))
    else:
        egamma_modifications = cms.VPSet() #reset all the modifications which so far are just VID
    egamma_modifications.append(makeEnergyScaleAndSmearingSysModifier("calibratedPatElectrons","calibratedPatPhotons"))
    egamma_modifications.append(egamma8XLegacyEtScaleSysModifier)

    #add the HEEP trk isol to the slimmed electron
    if runVID:
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
                           era="2017-Nov17ReReco",
                           eleIDModules=_defaultEleIDModules,
                           phoIDModules=_defaultPhoIDModules,
                           runVID=True,
                           applyEPCombBug=False):

    from PhysicsTools.SelectorUtils.tools.vid_id_tools import switchOnVIDElectronIdProducer,switchOnVIDPhotonIdProducer,setupAllVIDIdsInModule,DataFormat,setupVIDElectronSelection,setupVIDPhotonSelection
    # turn on VID producer, indicate data format  to be
    # DataFormat.AOD or DataFormat.MiniAOD, as appropriate
    if runVID:
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
        _setupEgammaPostRECOSequenceMiniAOD(process,applyEnergyCorrections=applyEnergyCorrections,applyVIDOnCorrectedEgamma=applyVIDOnCorrectedEgamma,era=era,runVID=runVID,applyEPCombBug=applyEPCombBug)
    else:
        _setupEgammaPostRECOSequence(process,applyEnergyCorrections=applyEnergyCorrections,applyVIDOnCorrectedEgamma=applyVIDOnCorrectedEgamma,era=era,runVID=runVID,applyEPCombBug=applyEPCombBug)
    
    process.egammaScaleSmearSeq = cms.Sequence( process.egammaScaleSmearTask)
    process.egammaPostRecoSeq   = cms.Sequence( process.egammaScaleSmearSeq)
    if runVID:
        process.egammaPostRecoSeq   = cms.Sequence( process.egammaScaleSmearSeq*
                                                    process.egmGsfElectronIDSequence*
                                                    process.egmPhotonIDSequence )

    return process
