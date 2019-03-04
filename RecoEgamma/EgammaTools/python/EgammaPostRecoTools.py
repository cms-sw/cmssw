
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
_fall17V2PhoMVAIDModules = [
    'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V2_cff'
    ]
_fall17V2PhoCutIDModules = [
    'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Fall17_94X_V2_cff'
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
    print "EgammaPostRecoTools: Fall17V2 electron modules not found, running ID without them. If you want Fall17V2 IDs, please merge the approprate PR\n  94X:  git cms-merge-topic cms-egamma/EgammaID_949"

if pkgutil.find_loader(_fall17V2PhoMVAIDModules[0]) != None:
    _defaultPhoIDModules.extend(_fall17V2PhoMVAIDModules)
else:
    print "EgammaPostRecoTools: Fall17V2 MVA photon modules not found, running ID without them. If you want Fall17V2 MVA Photon IDs, please merge the approprate PR\n  94X:  git cms-merge-topic cms-egamma/EgammaID_949\n  102X: git cms-merge-topic cms-egamma/EgammaID_1023"

if pkgutil.find_loader(_fall17V2PhoCutIDModules[0]) != None:
    _defaultPhoIDModules.extend(_fall17V2PhoCutIDModules)
else:
    print "EgammaPostRecoTools: Fall17V2 cut based Photons ID modules not found, running ID without them. If you want Fall17V2 CutBased Photon IDs, please merge the approprate PR\n  94X:  git cms-merge-topic cms-egamma/EgammaID_949\n  102X: git cms-merge-topic cms-egamma/EgammaID_1023"

def _getEnergyCorrectionFile(era):
    if era=="2017-Nov17ReReco":
        return "EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2017_17Nov2017_v1_ele_unc"
    if era=="2016-Legacy":
        return "EgammaAnalysis/ElectronTools/data/ScalesSmearings/Legacy2016_07Aug2017_FineEtaR9_v3_ele_unc"
    if era=="2016-Feb17ReMiniAOD":
        raise RuntimeError('Error in postRecoEgammaTools, era 2016-Feb17ReMiniAOD is not currently implimented') 
    if era=="2018-Prompt":
        raise RuntimeError('Error in postRecoEgammaTools, era 2018-Prompt does not have energy corrections availible yet, runEnergyCorrections must be set to false') 
    raise RuntimeError('Error in postRecoEgammaTools, era '+era+' not recognised. Allowed eras are 2017-Nov17ReReco, 2016-Legacy, 2016-Feb17ReMiniAOD')

def _is80XRelease(era):
    if era=="2016-Legacy" or era=="2016-Feb17ReMiniAOD": return True
    elif era!="2017-Nov17ReReco" and era!="2018-Prompt":
        raise RuntimeError('Error in postRecoEgammaTools, era '+era+' not recognised. Allowed eras are 2017-Nov17ReReco, 2016-Legacy, 2016-Feb17ReMiniAOD. 2018-Prompt')

def _getMVAsBeingRun(vidMod):
    mvasBeingRun = []
    for id_ in vidMod.physicsObjectIDs:
        for cut in id_.idDefinition.cutFlow:
            if cut.cutName.value().startswith("GsfEleMVA") or cut.cutName.value().startswith("PhoMVA"):
                mvaValueName = cut.mvaValueMapName.getProductInstanceLabel().replace("RawValues","Values")
                
                mvasBeingRun.append({'val' : {'prod' : cut.mvaValueMapName.getModuleLabel(),'name' : mvaValueName}, 'cat' : {'prod' : cut.mvaCategoriesMapName.getModuleLabel(),'name' : cut.mvaCategoriesMapName.getProductInstanceLabel() }})
    return mvasBeingRun
                
def _addMissingMVAValuesToUserData(process,egmod):

    if len(egmod)<2 or egmod[0].modifierName.value()!='EGExtraInfoModifierFromFloatValueMaps' or egmod[1].modifierName.value()!='EGExtraInfoModifierFromIntValueMaps':
        raise RuntimeError('dumping offending module {}\nError in postRecoEgammaTools._addMissingMVAValuesToUserData, we assume that the egamma_modifiers are setup so first its the float mod and then the int mod, this is currently not the case, the offending module dump is above'.format(egmod.dumpPython()))
    
    eleMVAs = _getMVAsBeingRun(process.egmGsfElectronIDs)
    phoMVAs = _getMVAsBeingRun(process.egmPhotonIDs)

    addVar = lambda modifier,var: setattr(modifier,var['name'],cms.InputTag(var['prod'],var['name']))
    
    for eleMVA in eleMVAs:
        if not hasattr(egmod[0].electron_config,eleMVA['val']['name']):
            addVar(egmod[0].electron_config,eleMVA['val'])
            addVar(egmod[1].electron_config,eleMVA['cat'])
    
    for phoMVA in phoMVAs:
        if not hasattr(egmod[0].photon_config,phoMVA['val']['name']):
            addVar(egmod[0].photon_config,phoMVA['val'])
            addVar(egmod[1].photon_config,phoMVA['cat'])
            


def _setupEgammaEnergyCorrections(process,eleSrc=cms.InputTag('gedGsfElectrons'),phoSrc=cms.InputTag('gedPhotons'),applyEnergyCorrections=False,era="2017-Nov17ReReco",runEnergyCorrections=True,applyEPCombBug=False):
    """ creates the AOD modules to run the energy corrections """

    if runEnergyCorrections == False:
        #didnt request energy corrections 
        process.egammaScaleSmearTask = cms.Task()
        return 
    
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

    energyCorrectionFile = _getEnergyCorrectionFile(era)
    process.calibratedElectrons.correctionFile = energyCorrectionFile
    process.calibratedPhotons.correctionFile = energyCorrectionFile

    
    if applyEPCombBug and hasattr(process.calibratedPatElectrons,"useSmearCorrEcalEnergyErrInComb"):
        process.calibratedPatElectrons.useSmearCorrEcalEnergyErrInComb=True
    elif hasattr(process.calibratedPatElectrons,"useSmearCorrEcalEnergyErrInComb"):
        process.calibratedPatElectrons.useSmearCorrEcalEnergyErrInComb=False
    elif applyEPCombBug:
        raise RuntimeError('Error in postRecoEgammaTools, the E/p combination bug can not be applied in >= 10_2_X (applyEPCombBug must be False), it is only possible to emulate in 9_4_X')

def _setupEgammaPostRECOSequence(process,applyEnergyCorrections=False,applyVIDOnCorrectedEgamma=False,era="2017-Nov17ReReco",runVID=True,runEnergyCorrections=True,applyEPCombBug=False):
    
    if runVID==False and runEnergyCorrections==False:
        #nothing to actually do, just return
        process.egammaScaleSmearTask = cms.Task()
        return

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


    _setupEgammaEnergyCorrections(process,eleSrc=eleSrc,phoSrc=phoSrc,applyEnergyCorrections=applyEnergyCorrections,era=era,runEnergyCorrections=runEnergyCorrections,applyEPCombBug=applyEPCombBug)

    if runVID:
        process.egmGsfElectronIDs.physicsObjectSrc = eleSrc
        process.egmPhotonIDs.physicsObjectSrc = phoSrc
        process.electronMVAValueMapProducer.src = eleSrc  
        if hasattr(process,'electronMVAVariableHelper'):
            process.electronMVAVariableHelper.srcMiniAOD = eleSrc
        process.photonMVAValueMapProducer.src = phoSrc
        process.photonIDValueMapProducer.src = phoSrc
        process.egmPhotonIsolation.srcToIsolate = phoSrc
    
    if runVID and hasattr(process,'heepIDVarValueMaps'):
        process.heepIDVarValueMaps.elesAOD = eleSrc
        process.heepIDVarValueMaps.dataFormat = 1
    if hasattr(process,'packedCandsForTkIso') and era.find("2016")!=-1:
        process.packedCandsForTkIso.chargedHadronIsolation = ""
        
                                          
def _setupEgammaEnergyCorrectionsMiniAOD(process,eleSrc,phoSrc,applyEnergyCorrections=False,applyVIDOnCorrectedEgamma=False,era="2017-Nov17ReReco",runEnergyCorrections=True,applyEPCombBug=False):
    """sets up the e/gamma energy corrections for miniAOD
    it will adjust eleSrc and phoSrc to the correct values
    """

    if runEnergyCorrections == False:
        return

    process.load('RecoEgamma.EgammaTools.calibratedEgammas_cff')
    #we copy the input tag as we may change them later
    process.calibratedPatElectrons.src = cms.InputTag(eleSrc.value())
    process.calibratedPatPhotons.src = cms.InputTag(phoSrc.value())
    
    energyCorrectionFile = _getEnergyCorrectionFile(era)
    process.calibratedPatElectrons.correctionFile = energyCorrectionFile
    process.calibratedPatPhotons.correctionFile = energyCorrectionFile

    if applyEPCombBug and hasattr(process.calibratedPatElectrons,'useSmearCorrEcalEnergyErrInComb'):
        process.calibratedPatElectrons.useSmearCorrEcalEnergyErrInComb=True
    elif hasattr(process.calibratedPatElectrons,'useSmearCorrEcalEnergyErrInComb'):
        process.calibratedPatElectrons.useSmearCorrEcalEnergyErrInComb=False
    elif applyEPCombBug:
        raise RuntimeError('Error in postRecoEgammaTools, the E/p combination bug can not be applied in >= 10_2_X (applyEPCombBug must be False) , it is only possible to emulate in 9_4_X')

    if applyEnergyCorrections or applyVIDOnCorrectedEgamma:
        process.calibratedPatElectrons.produceCalibratedObjs = True
        process.calibratedPatPhotons.produceCalibratedObjs = True
    else:
        process.calibratedPatElectrons.produceCalibratedObjs = False 
        process.calibratedPatPhotons.produceCalibratedObjs = False 

    


def _setupEgammaPostRECOSequenceMiniAOD(process,applyEnergyCorrections=False,applyVIDOnCorrectedEgamma=False,era="2017-Nov17ReReco",runVID=True,runEnergyCorrections=True,applyEPCombBug=False):
    """
    This function loads the calibrated producers calibratedPatElectrons,calibratedPatPhotons, 
    sets VID & other modules to the correct electron/photon source,
    loads up the modifiers and which then creates a new slimmedElectrons,slimmedPhotons collection
    with VID and scale and smearing all loaded in
    """
    
    if applyEnergyCorrections != applyVIDOnCorrectedEgamma:
        raise RuntimeError('Error, applyEnergyCorrections {} and applyVIDOnCorrectedEgamma {} must be equal to each other for now,\n functionality for them to be different isnt yet availible'.format(applyEnergyCorrections,applyVIDOnCorrectedEgamma))


    phoSrc = cms.InputTag('slimmedPhotons',processName=cms.InputTag.skipCurrentProcess())
    eleSrc = cms.InputTag('slimmedElectrons',processName=cms.InputTag.skipCurrentProcess())
    phoCalibSrc = cms.InputTag('calibratedPatPhotons')
    eleCalibSrc = cms.InputTag('calibratedPatElectrons')

    _setupEgammaEnergyCorrectionsMiniAOD(process,eleSrc=eleSrc,phoSrc=phoSrc,applyEnergyCorrections=applyEnergyCorrections,applyVIDOnCorrectedEgamma=applyVIDOnCorrectedEgamma,era=era,runEnergyCorrections=runEnergyCorrections,applyEPCombBug=applyEPCombBug)

    if applyVIDOnCorrectedEgamma:
        phoVIDSrc = phoCalibSrc
        eleVIDSrc = eleCalibSrc
    else:
        phoVIDSrc = phoSrc
        eleVIDSrc = eleSrc

    if applyEnergyCorrections:
        phoNewSrc = phoCalibSrc
        eleNewSrc = eleCalibSrc
    else:
        phoNewSrc = phoSrc
        eleNewSrc = eleSrc

    if runVID:
        process.egmGsfElectronIDs.physicsObjectSrc = eleVIDSrc
        process.egmPhotonIDs.physicsObjectSrc = phoVIDSrc
        process.electronMVAValueMapProducer.srcMiniAOD = eleVIDSrc
        if hasattr(process,'electronMVAVariableHelper'):
            process.electronMVAVariableHelper.srcMiniAOD = eleVIDSrc
        process.photonMVAValueMapProducer.srcMiniAOD = phoVIDSrc
        process.photonIDValueMapProducer.srcMiniAOD = phoVIDSrc
        process.egmPhotonIsolation.srcToIsolate = phoVIDSrc

    if runVID and hasattr(process,'heepIDVarValueMaps'):
        process.heepIDVarValueMaps.elesMiniAOD = eleVIDSrc
        process.heepIDVarValueMaps.dataFormat = 2


    from RecoEgamma.EgammaTools.egammaObjectModificationsInMiniAOD_cff import egamma_modifications,egamma8XLegacyEtScaleSysModifier,egamma8XObjectUpdateModifier
    from RecoEgamma.EgammaTools.egammaObjectModifications_tools import makeVIDBitsModifier,makeVIDinPATIDsModifier,makeEnergyScaleAndSmearingSysModifier  
    if runVID:
        egamma_modifications.append(makeVIDBitsModifier(process,"egmGsfElectronIDs","egmPhotonIDs"))
        egamma_modifications.append(makeVIDinPATIDsModifier(process,"egmGsfElectronIDs","egmPhotonIDs"))
    else:
        egamma_modifications = cms.VPSet() #reset all the modifications which so far are just VID
    if _is80XRelease(era): 
        egamma_modifications.append(egamma8XObjectUpdateModifier) #if we were generated in 80X, we need fill in missing data members in 94X
    if runEnergyCorrections:
        egamma_modifications.append(makeEnergyScaleAndSmearingSysModifier("calibratedPatElectrons","calibratedPatPhotons"))
        egamma_modifications.append(egamma8XLegacyEtScaleSysModifier)


    #add any missing variables to the slimmed electron 
    if runVID:
        #MVA V2 values may not be added by default due to data format consistency issues
        _addMissingMVAValuesToUserData(process,egamma_modifications)
        #now add HEEP trk isolation
        for pset in egamma_modifications:
            if pset.hasParameter("modifierName") and pset.modifierName == cms.string('EGExtraInfoModifierFromFloatValueMaps'):
                pset.electron_config.heepTrkPtIso = cms.InputTag("heepIDVarValueMaps","eleTrkPtIso")
                break

    for pset in egamma_modifications:
        pset.overrideExistingValues = cms.bool(True)
        if hasattr(pset,"electron_config"): pset.electron_config.electronSrc = eleNewSrc
        if hasattr(pset,"photon_config"): pset.photon_config.photonSrc = phoNewSrc

    process.slimmedElectrons = cms.EDProducer("ModifiedElectronProducer",
                                              src=eleNewSrc,
                                              modifierConfig = cms.PSet(
                                                  modifications = egamma_modifications
                                                  )
                                              )
    process.slimmedPhotons = cms.EDProducer("ModifiedPhotonProducer",
                                            src=phoNewSrc,
                                            modifierConfig = cms.PSet(
                                                modifications = egamma_modifications
                                                )
                                            )

    process.egammaScaleSmearTask = cms.Task()
    process.egammaPostRecoPatUpdatorTask = cms.Task()
    #we only run if the modifications are going to do something
    if egamma_modifications != cms.VPSet():
        process.egammaPostRecoPatUpdatorTask.add(process.slimmedElectrons)
        process.egammaPostRecoPatUpdatorTask.add(process.slimmedPhotons)
        if runEnergyCorrections:
            process.egammaScaleSmearTask.add(process.calibratedPatElectrons)
            process.egammaScaleSmearTask.add(process.calibratedPatPhotons)


def setupEgammaPostRecoSeq(process,
                           applyEnergyCorrections=False,
                           applyVIDOnCorrectedEgamma=False,
                           isMiniAOD=True,
                           era="2017-Nov17ReReco",
                           eleIDModules=_defaultEleIDModules,
                           phoIDModules=_defaultPhoIDModules,
                           runVID=True,
                           runEnergyCorrections=True,
                           applyEPCombBug=False,
                           autoAdjustParams=True):

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

    if autoAdjustParams:
        if era == "2018-Prompt" and runEnergyCorrections: 
            print "EgammaPostRecoTools:\n  2018-Prompt does not yet have residual scales and smearings availible, setting runEnergyCorrections to False. To override, set autoAdjustParams = False"
            runEnergyCorrections=False

    if isMiniAOD:
        _setupEgammaPostRECOSequenceMiniAOD(process,applyEnergyCorrections=applyEnergyCorrections,applyVIDOnCorrectedEgamma=applyVIDOnCorrectedEgamma,era=era,runVID=runVID,runEnergyCorrections=runEnergyCorrections,applyEPCombBug=applyEPCombBug)
    else:
        _setupEgammaPostRECOSequence(process,applyEnergyCorrections=applyEnergyCorrections,applyVIDOnCorrectedEgamma=applyVIDOnCorrectedEgamma,era=era,runVID=runVID,runEnergyCorrections=runEnergyCorrections,applyEPCombBug=applyEPCombBug)
    
    process.egammaScaleSmearSeq = cms.Sequence(process.egammaScaleSmearTask)
    #post reco seq is calibrations -> vid -> pat updator 
    process.egammaPostRecoSeq   = cms.Sequence(process.egammaScaleSmearSeq)
    if not runEnergyCorrections and runVID:
        process.egammaPostRecoSeq = cms.Sequence(process.egmGsfElectronIDSequence*process.egmPhotonIDSequence)
    elif runVID:
        process.egammaPostRecoSeq.insert(-1,process.egmGsfElectronIDSequence)
        process.egammaPostRecoSeq.insert(-1,process.egmPhotonIDSequence)
    if isMiniAOD:
        process.egammaPostRecoPatUpdatorSeq = cms.Sequence(process.egammaPostRecoPatUpdatorTask)
        process.egammaPostRecoSeq.insert(-1,process.egammaPostRecoPatUpdatorSeq)     
                       
    return process
