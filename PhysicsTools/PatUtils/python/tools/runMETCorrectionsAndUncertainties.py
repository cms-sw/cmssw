import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *
import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection


def isValidInputTag(input):
    input_str = input
    if isinstance(input, cms.InputTag):
        input_str = input.value()
    if input is None or input_str == '""':
        return False
    else:
        return True


class RunMETCorrectionsAndUncertainties(ConfigToolBase):
  
    _label='RunMETCorrectionsAndUncertainties'
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters, 'metType', "PF",
                          "Type of considered MET (only PF supported so far)", Type=str)
        self.addParameter(self._defaultParameters, 'correctionLevel', [""],
                          "level of correction : available corrections for pfMet are T0, T1, T2, Txy and Smear; irrelevant entry for MVAMet)",
                          allowedValues=["T0","T1","T2","Txy","Smear",""])
        self.addParameter(self._defaultParameters, 'computeUncertainties', True,
                          "enable/disable the uncertainty computation", Type=bool)
        self.addParameter(self._defaultParameters, 'produceIntermediateCorrections', False,
                          "enable/disable the production of all correction schemes (only for the most common)", Type=bool)
        self.addParameter(self._defaultParameters, 'electronCollection', cms.InputTag('selectedPatElectrons'),
	                  "Input electron collection", Type=cms.InputTag, acceptNoneValue=True)
#  empty default InputTag for photons to avoid double-counting wrt. cleanPatElectrons collection
	self.addParameter(self._defaultParameters, 'photonCollection', None,
	                  "Input photon collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'muonCollection', cms.InputTag('selectedPatMuons'),
                          "Input muon collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'tauCollection', cms.InputTag('selectedPatTaus'),
                          "Input tau collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'jetCollectionUnskimmed', cms.InputTag('patJets'),
                          "Input unskimmed jet collection for T1 MET computation", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'pfCandCollection', cms.InputTag('particleFlow'),
                          "pf Candidate collection", Type=cms.InputTag, acceptNoneValue=True)
        self.addParameter(self._defaultParameters, 'autoJetCleaning', 'LepClean',
                          "Enable the jet cleaning for the uncertainty computation: Full for tau/photons/jet cleaning, Partial for jet cleaning, LepClean for jet cleaning with muon and electrons only, None or Manual for no cleaning", Type=str)
        self.addParameter(self._defaultParameters, 'jetFlavor', 'AK4PFchs',
                          "Use AK4PF/AK4PFchs for PFJets,AK4Calo for CaloJets", Type=str)
        self.addParameter(self._defaultParameters, 'jetCorrectionType', 'L1L2L3-L1',
                          "Use L1L2L3-L1 for the standard L1 removal / L1L2L3-RC for the random-cone correction", Type=str)

        self.addParameter(self._defaultParameters, 'jetCorLabelUpToL3', "ak4PFCHSL1FastL2L3Corrector", "Use ak4PFL1FastL2L3Corrector (ak4PFCHSL1FastL2L3Corrector) for PFJets with (without) charged hadron subtraction, ak4CaloL1FastL2L3Corrector for CaloJets", Type=str)
        self.addParameter(self._defaultParameters, 'jetCorLabelL3Res', "ak4PFCHSL1FastL2L3ResidualCorrector", "Use ak4PFL1FastL2L3ResidualCorrector (ak4PFCHSL1FastL2L3ResidualCorrector) for PFJets with (without) charged hadron subtraction, ak4CaloL1FastL2L3ResidualCorrector for CaloJets", Type=str)

# the file is used only for local running
        self.addParameter(self._defaultParameters, 'jecUncertaintyFile', '',
                          "Extra JES uncertainty file", Type=str)
        self.addParameter(self._defaultParameters, 'jecUncertaintyTag', 'Uncertainty',
                          "JES uncertainty Tag", Type=str)
        
        self.addParameter(self._defaultParameters, 'mvaMetLeptons',["Electrons","Muons"],
                          "Leptons to be used for recoil computation in the MVA MET, available values are: Electrons, Muons, Taus, Photons", allowedValues=["Electrons","Muons","Taus","Photons",""])

        self.addParameter(self._defaultParameters, 'addToPatDefaultSequence', False,
                          "Flag to enable/disable that metUncertaintySequence is inserted into patDefaultSequence", Type=bool)
        self.addParameter(self._defaultParameters, 'manualJetConfig', False,
                  "Enable jet configuration options", Type=bool)
	self.addParameter(self._defaultParameters, 'jetSelection', 'pt>15 && abs(eta)<9.9',
                          "Advanced jet kinematic selection", Type=str)
        self.addParameter(self._defaultParameters, 'recoMetFromPFCs', False,
                  "Recompute the MET from scratch using the pfCandidate collection", Type=bool)
        self.addParameter(self._defaultParameters, 'reapplyJEC', True,
                  "Flag to enable/disable JEC update", Type=bool)
        self.addParameter(self._defaultParameters, 'reclusterJets', False,
                  "Flag to enable/disable the jet reclustering", Type=bool)
        self.addParameter(self._defaultParameters, 'CHS', False,
                  "Flag to enable/disable the CHS jets", Type=bool)
        self.addParameter(self._defaultParameters, 'runOnData', False,
                          "Switch for data/MC processing", Type=bool)
        self.addParameter(self._defaultParameters, 'onMiniAOD', False,
                          "Switch on miniAOD configuration", Type=bool)
        self.addParameter(self._defaultParameters, 'postfix', '',
                          "Technical parameter to identify the resulting sequence and its modules (allows multiple calls in a job)", Type=str)
        self._parameters = copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

#=========================================================================================
    def __call__(self, process,
                 metType                 =None,
                 correctionLevel         =None,
                 computeUncertainties    =None,
                 produceIntermediateCorrections = None,
                 electronCollection      =None,
                 photonCollection        =None,
                 muonCollection          =None,
                 tauCollection           =None,
                 jetCollectionUnskimmed  =None,
                 pfCandCollection        =None,
                 autoJetCleaning         =None,
                 jetFlavor               =None,
                 jetCorr                 =None,
                 jetCorLabelUpToL3       =None,
                 jetCorLabelL3Res        =None,
                 jecUncertaintyFile      =None,
                 jecUncertaintyTag       =None,
                 mvaMetLeptons           =None,
                 addToPatDefaultSequence =None,
                 manualJetConfig         =None,
                 jetSelection            =None,
                 recoMetFromPFCs         =None,
                 reapplyJEC              =None,
                 reclusterJets           =None,
                 CHS                     =None,
                 runOnData               =None,
                 onMiniAOD               =None,
                 postfix                 =None):
        electronCollection = self.initializeInputTag(electronCollection, 'electronCollection')
        photonCollection = self.initializeInputTag(photonCollection, 'photonCollection')
        muonCollection = self.initializeInputTag(muonCollection, 'muonCollection')
        tauCollection = self.initializeInputTag(tauCollection, 'tauCollection')
        jetCollectionUnskimmed = self.initializeInputTag(jetCollectionUnskimmed, 'jetCollectionUnskimmed')
        pfCandCollection = self.initializeInputTag(pfCandCollection, 'pfCandCollection')
        if metType is None :
            metType =  self._defaultParameters['metType'].value
        if correctionLevel is None :
            correctionLevel = self._defaultParameters['correctionLevel'].value
        if computeUncertainties is None :
            computeUncertainties = self._defaultParameters['computeUncertainties'].value
        if produceIntermediateCorrections is None :
            produceIntermediateCorrections = self._defaultParameters['produceIntermediateCorrections'].value
        if electronCollection is None :
            electronCollection = self._defaultParameters['electronCollection'].value
        if photonCollection is None :
            photonCollection = self._defaultParameters['photonCollection'].value
        if muonCollection is None :
            muonCollection = self._defaultParameters['muonCollection'].value
        if tauCollection is None :
            tauCollection = self._defaultParameters['tauCollection'].value
        if jetCollectionUnskimmed is None :
            jetCollectionUnskimmed = self._defaultParameters['jetCollectionUnskimmed'].value
        if pfCandCollection is None :
            pfCandCollection = self._defaultParameters['pfCandCollection'].value
        if autoJetCleaning is None :
            autoJetCleaning = self._defaultParameters['autoJetCleaning'].value
        if jetFlavor is None :
            jetFlavor = self._defaultParameters['jetFlavor'].value
        if jetCorr is None :
            jetCorr = self._defaultParameters['jetCorrectionType'].value
        if jetCorLabelUpToL3  is None:
            jetCorLabelUpToL3 = self._defaultParameters['jetCorLabelUpToL3'].value
        if jetCorLabelL3Res   is None:
            jetCorLabelL3Res = self._defaultParameters['jetCorLabelL3Res'].value
        if jecUncertaintyFile is None:
            jecUncertaintyFile = self._defaultParameters['jecUncertaintyFile'].value
        if jecUncertaintyTag  is None:
            jecUncertaintyTag = self._defaultParameters['jecUncertaintyTag'].value
            
        if mvaMetLeptons is None:
            mvaMetLeptons = self._defaultParameters['mvaMetLeptons'].value

        if addToPatDefaultSequence is None :
            addToPatDefaultSequence = self._defaultParameters['addToPatDefaultSequence'].value
        if manualJetConfig is None :
            manualJetConfig =  self._defaultParameters['manualJetConfig'].value
        if jetSelection is None :
            jetSelection = self._defaultParameters['jetSelection'].value
        if recoMetFromPFCs is None :
            recoMetFromPFCs =  self._defaultParameters['recoMetFromPFCs'].value
        if reapplyJEC is None :
            reapplyJEC = self._defaultParameters['reapplyJEC'].value
        if reclusterJets is None :
            reclusterJets = self._defaultParameters['reclusterJets'].value
        if CHS is None :
            CHS = self._defaultParameters['CHS'].value
        if runOnData is None :
            runOnData = self._defaultParameters['runOnData'].value
        if onMiniAOD is None :
            onMiniAOD = self._defaultParameters['onMiniAOD'].value
        if postfix is None :
            postfix = self._defaultParameters['postfix'].value

        self.setParameter('metType',metType),
        self.setParameter('correctionLevel',correctionLevel),
        self.setParameter('computeUncertainties',computeUncertainties),
        self.setParameter('produceIntermediateCorrections',produceIntermediateCorrections),
        self.setParameter('electronCollection',electronCollection),
        self.setParameter('photonCollection',photonCollection),
        self.setParameter('muonCollection',muonCollection),
        self.setParameter('tauCollection',tauCollection),
        self.setParameter('jetCollectionUnskimmed',jetCollectionUnskimmed),
        self.setParameter('pfCandCollection',pfCandCollection),

        self.setParameter('autoJetCleaning',autoJetCleaning),
        self.setParameter('jetFlavor',jetFlavor),

        #optional
        self.setParameter('jecUncertaintyFile',jecUncertaintyFile),
        self.setParameter('jecUncertaintyTag',jecUncertaintyTag),

        self.setParameter('mvaMetLeptons',mvaMetLeptons),
        
        self.setParameter('addToPatDefaultSequence',addToPatDefaultSequence),
        self.setParameter('jetSelection',jetSelection),
        self.setParameter('recoMetFromPFCs',recoMetFromPFCs),
        self.setParameter('runOnData',runOnData),
        self.setParameter('onMiniAOD',onMiniAOD),
        self.setParameter('postfix',postfix),

        #if mva/puppi MET, autoswitch to std jets
        if metType == "MVA" or 'Puppi' in postfix:
            self.setParameter('CHS',False),

        #jet energy scale uncertainty needs
        if manualJetConfig:
            self.setParameter('CHS',CHS)
            self.setParameter('jetCorLabelUpToL3',jetCorLabelUpToL3)
            self.setParameter('jetCorLabelL3Res',jetCorLabelL3Res)
            self.setParameter('reclusterJets',reclusterJets)
        else:
             #internal jet configuration
            self.jetConfiguration()
        
        #met reprocessing and jet reclustering
        if recoMetFromPFCs:
            self.setParameter('reclusterJets',True)
        
        #ZD: puppi jet reclustering breaks the puppi jets
        #overwriting of jet reclustering parameter for puppi
        if 'Puppi' in postfix and not onMiniAOD:
            self.setParameter('reclusterJets',False)

        #jet collection overloading for automatic jet reclustering or JEC application
        if reclusterJets:
            self.setParameter('jetCollectionUnskimmed',cms.InputTag('patJets'))
            
        self.apply(process)
        

    def toolCode(self, process):
        metType                 = self._parameters['metType'].value
        correctionLevel         = self._parameters['correctionLevel'].value
        computeUncertainties    = self._parameters['computeUncertainties'].value
        produceIntermediateCorrections = self._parameters['produceIntermediateCorrections'].value
        electronCollection      = self._parameters['electronCollection'].value
        photonCollection        = self._parameters['photonCollection'].value
        muonCollection          = self._parameters['muonCollection'].value
        tauCollection           = self._parameters['tauCollection'].value
        jetCollectionUnskimmed  = self._parameters['jetCollectionUnskimmed'].value
        pfCandCollection        = self._parameters['pfCandCollection'].value
        autoJetCleaning         = self._parameters['autoJetCleaning'].value
        jetFlavor               = self._parameters['jetFlavor'].value
        jetCorLabelUpToL3       = self._parameters['jetCorLabelUpToL3'].value
        jetCorLabelL3Res        = self._parameters['jetCorLabelL3Res'].value
        jecUncertaintyFile      = self._parameters['jecUncertaintyFile'].value
        jecUncertaintyTag       = self._parameters['jecUncertaintyTag'].value

        mvaMetLeptons           = self._parameters['mvaMetLeptons'].value
        addToPatDefaultSequence = self._parameters['addToPatDefaultSequence'].value
        jetSelection            = self._parameters['jetSelection'].value
        recoMetFromPFCs         = self._parameters['recoMetFromPFCs'].value
        reapplyJEC              = self._parameters['reapplyJEC'].value
        reclusterJets           = self._parameters['reclusterJets'].value
        onMiniAOD               = self._parameters['onMiniAOD'].value
        postfix                 = self._parameters['postfix'].value
        
        #prepare jet configuration
        jetUncInfos = { "jCorrPayload":jetFlavor, "jCorLabelUpToL3":jetCorLabelUpToL3,
                        "jCorLabelL3Res":jetCorLabelL3Res, "jecUncFile":jecUncertaintyFile,
                        "jecUncTag":jecUncertaintyTag }        

        patMetModuleSequence = cms.Sequence()

        # recompute the MET (and thus the jets as well for correction) from scratch
        if recoMetFromPFCs:
            self.recomputeRawMetFromPfcs(process, 
                                         pfCandCollection, 
                                         onMiniAOD,
                                         patMetModuleSequence,
                                         postfix)
            
        elif onMiniAOD: #raw MET extraction if running on miniAODs
            self.extractMET(process, "raw", patMetModuleSequence, postfix)


        #default MET production
        self.produceMET(process, metType,patMetModuleSequence, postfix)
        
            
        #jet AK4 reclustering if needed for JECs
        if reclusterJets:
            jetCollectionUnskimmed = self.ak4JetReclustering(process, pfCandCollection, 
                                                             patMetModuleSequence, postfix)

        # or reapplication of jecs
        if onMiniAOD:
            if not reclusterJets and reapplyJEC:
                jetCollectionUnskimmed = self.updateJECs(process, jetCollectionUnskimmed, patMetModuleSequence, postfix)

        #getting the jet collection that will be used for corrections 
        #and uncertainty computation
        jetCollection = self.getJetCollectionForCorsAndUncs(process, 
                                                            jetCollectionUnskimmed,
                                                            jetSelection,
                                                            autoJetCleaning,
                                                            patMetModuleSequence,
                                                            postfix)
        #preparation to run over miniAOD (met reproduction) 
        if onMiniAOD:
            self.miniAODConfiguration(process, 
                                      pfCandCollection,
                                      jetCollection,
                                      patMetModuleSequence,
                                      postfix
                                      )        

        # correct the MET
        patMetCorrectionSequence, metModName = self.getCorrectedMET(process, metType, correctionLevel,
                                                                    produceIntermediateCorrections,
                                                                    jetCollection,
                                                                    patMetModuleSequence, postfix )
        
        #fix the default jets for the type1 computation to those used to compute the uncertainties
        #in order to be consistent with what is done in the correction and uncertainty step
        #particularly true for miniAODs
        if "T1" in metModName:
            getattr(process,"patPFMetT1T2Corr"+postfix).src = cms.InputTag(jetCollection.value()+postfix)
            getattr(process,"patPFMetT2Corr"+postfix).src = cms.InputTag(jetCollection.value()+postfix)
            #ZD:puppi currently doesn't have the L1 corrections in the GT
            if 'Puppi' in postfix:
                getattr(process,"patPFMetT1T2Corr"+postfix).offsetCorrLabel = cms.InputTag("")
                getattr(process,"patPFMetT2Corr"+postfix).offsetCorrLabel = cms.InputTag("")
        if "Smear" in metModName:
            getattr(process,"patSmearedJets"+postfix).src = cms.InputTag(jetCollection.value()+postfix)
            if 'Puppi' in postfix:
                getattr(process,"patPFMetT1T2SmearCorr"+postfix).offsetCorrLabel = cms.InputTag("")

 
        #compute the uncertainty on the MET
        patMetUncertaintySequence = cms.Sequence()
        patShiftedModuleSequence = cms.Sequence()
        if computeUncertainties:
            patMetUncertaintySequence,patShiftedModuleSequence =  self.getMETUncertainties(process, metType, metModName,
                                                                  electronCollection,
                                                                  photonCollection,
                                                                  muonCollection,
                                                                  tauCollection,
                                                                  pfCandCollection,
                                                                  jetCollection,
                                                                  jetUncInfos,
                                                                  postfix)
        
        if not hasattr(process, "patMetCorrectionSequence"+postfix):
            setattr(process, "patMetCorrectionSequence"+postfix, patMetCorrectionSequence)
        if not hasattr(process, "patMetUncertaintySequence"+postfix):
            setattr(process, "patMetUncertaintySequence"+postfix, patMetUncertaintySequence)
        if not hasattr(process, "patShiftedModuleSequence"+postfix):
            setattr(process, "patShiftedModuleSequence"+postfix, patShiftedModuleSequence)
        if not hasattr(process, "patMetModuleSequence"+postfix):
            setattr(process, "patMetModuleSequence"+postfix, patMetModuleSequence)
        
        #prepare and fill the final sequence containing all the sub-sequence
        fullPatMetSequence = cms.Sequence()
        fullPatMetSequence += getattr(process, "patMetModuleSequence"+postfix)
        fullPatMetSequence += getattr(process, "patMetUncertaintySequence"+postfix)
        fullPatMetSequence += getattr(process, "patShiftedModuleSequence"+postfix)
        fullPatMetSequence += getattr(process, "patMetCorrectionSequence"+postfix)
        
        setattr(process,"fullPatMetSequence"+postfix,fullPatMetSequence)

        # insert the fullPatMetSequence into patDefaultSequence if needed
        if addToPatDefaultSequence:
            if not hasattr(process, "patDefaultSequence"):
                raise ValueError("PAT default sequence is not defined !!")
            process.patDefaultSequence += getattr(process, "fullPatMetSequence"+postfix)
    
#====================================================================================================
    def produceMET(self, process,  metType, metModuleSequence, postfix):
        if metType == "PF" and not hasattr(process, 'pat'+metType+'Met'):
            process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")
            
        if postfix != "" and metType == "PF" and not hasattr(process, 'pat'+metType+'Met'+postfix):
            configtools.cloneProcessingSnippet(process, getattr(process,"producePatPFMETCorrections"), postfix)
            setattr(process, 'pat'+metType+'Met'+postfix, getattr(process,'patPFMet' ).clone() )
            getattr(process, "patPFMet"+postfix).metSource = cms.InputTag("pfMet"+postfix)
            getattr(process, "patPFMet"+postfix).srcPFCands = self._parameters["pfCandCollection"].value
        
        if self._parameters["runOnData"].value:
            getattr(process, "patPFMet"+postfix).addGenMET  = False
           
            
        #MM: FIXME MVA
        if metType == "MVA": # and not hasattr(process, 'pat'+metType+'Met'):
           # process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")
            mvaMetProducer = self.createMVAMETModule(process)
            setattr(process, 'pfMVAMet'+postfix, mvaMetProducer )
            setattr(process, 'pat'+metType+'Met'+postfix, getattr(process,'patPFMet' ).clone(
                    metSource = cms.InputTag('pfMVAMet'),
                    
                    ) ) 
            
        metModuleSequence += getattr(process, 'pat'+metType+'Met'+postfix )

#====================================================================================================
    def getCorrectedMET(self, process, metType, correctionLevel,produceIntermediateCorrections, 
                        jetCollection, metModuleSequence, postfix ):
        
        # default outputs
        patMetCorrectionSequence = cms.Sequence()
        metModName = "pat"+metType+"Met"+postfix
       
        if metType == "MVA": #corrections are irrelevant for the MVA MET (except jet smearing?)
            return patMetCorrectionSequence, metModName
                           
        corNames = { #not really needed but in case we have changes in the future....
            "T0":"T0pc",
            "T1":"T1",
            "T2":"T2",
            "Txy":"Txy",
            "Smear":"Smear",
            }
        
        
        #if empty correction level, no need to try something
        for cor in correctionLevel:
            if cor not in corNames.keys():
                if cor != "":
                    print "ERROR : ",cor," is not a proper MET correction name! aborting the MET correction production"
                return patMetCorrectionSequence, metModName

        corModNames = {
            "T0": "patPFMetT0CorrSequence"+postfix,
            "T1": "patPFMetT1T2CorrSequence"+postfix,
            "T2": "patPFMetT2CorrSequence"+postfix,
            "Txy": "patPFMetTxyCorrSequence"+postfix,
            "Smear": "patPFMetSmearCorrSequence"+postfix,
            "T2Smear": "patPFMetT2SmearCorrSequence"+postfix
            }

        if postfix != "":
            configtools.cloneProcessingSnippet(process, getattr(process,"patPFMetT0CorrSequence"), postfix)
            configtools.cloneProcessingSnippet(process, getattr(process,"patPFMetT1T2CorrSequence"), postfix)
            configtools.cloneProcessingSnippet(process, getattr(process,"patPFMetT2CorrSequence"), postfix)
            configtools.cloneProcessingSnippet(process, getattr(process,"patPFMetTxyCorrSequence"), postfix)
            configtools.cloneProcessingSnippet(process, getattr(process,"patPFMetSmearCorrSequence"), postfix)
            configtools.cloneProcessingSnippet(process, getattr(process,"patPFMetT2SmearCorrSequence"), postfix)
           
        corModules = {}
        for mod in corModNames.keys():
            corModules[mod] = getattr(process, corModNames[mod] )
                  
        corTags = {
            "T0":cms.InputTag('patPFMetT0Corr'+postfix),
            "T1":cms.InputTag('patPFMetT1T2Corr'+postfix, 'type1'),
            "T2":cms.InputTag('patPFMetT2Corr'+postfix,   'type2'),
            "Txy": cms.InputTag('patPFMetTxyCorr'+postfix),
            "Smear":cms.InputTag('patPFMetT1T2SmearCorr'+postfix, 'type1'),
            "T2Smear":cms.InputTag('patPFMetT2SmearCorr'+postfix, 'type2') 
            }

        corScheme=""
        corrections = []
        correctionSequence = []
        for cor in correctionLevel:
            corScheme += corNames[cor]
            corrections.append(corTags[cor])
            correctionSequence.append(corModules[cor])

        #T2 and smearing corModuleTag switch, specific case
        if "T2" in correctionLevel and "Smear" in correctionLevel:
            corrections.append(corTags["T2Smear"])
            correctionSequence.append(corModules["T2Smear"])

        #if both are here, consider smeared corJets for the full T1+Smear correction
        if "T1" in correctionLevel and "Smear" in correctionLevel:
            corrections.remove(corTags["T1"])
            
        #Txy parameter tuning
        if "Txy" in correctionLevel:
            self.tuneTxyParameters(process, corScheme, postfix)
            getattr(process, "patPFMetTxyCorr"+postfix).srcPFlow = self._parameters["pfCandCollection"].value
     
        #Enable MET significance if the type1 MET is computed
        if "T1" in correctionLevel:
            getattr(process, "pat"+metType+"Met"+postfix).computeMETSignificance = cms.bool(True)

        #T1 parameter tuning when CHS jets are not used
        if "T1" in correctionLevel and not self._parameters["CHS"].value:  
            setattr(process, "corrPfMetType1"+postfix, getattr(process, "corrPfMetType1" ).clone() )
            getattr(process, "corrPfMetType1"+postfix).src =  cms.InputTag("ak4PFJets"+postfix)
            getattr(process, "corrPfMetType1"+postfix).jetCorrLabel = cms.InputTag("ak4PFL1FastL2L3Corrector")
            getattr(process, "corrPfMetType1"+postfix).jetCorrLabelRes = cms.InputTag("ak4PFL1FastL2L3ResidualCorrector")
            getattr(process, "corrPfMetType1"+postfix).offsetCorrLabel = cms.InputTag("ak4PFL1FastjetCorrector")
            getattr(process, "basicJetsForMet"+postfix).offsetCorrLabel = cms.InputTag("ak4PFL1FastjetCorrector")
        
        if "T1" in correctionLevel and self._parameters["CHS"].value and self._parameters["reclusterJets"].value:
            getattr(process, "corrPfMetType1"+postfix).src =  cms.InputTag("ak4PFJetsCHS"+postfix)

        #create the main MET producer
        metModName = "pat"+metType+"Met"+corScheme+postfix

        sequenceName=""
        corMetProducer=None
        if metType == "PF":
            corMetProducer = cms.EDProducer("CorrectedPATMETProducer",
                       src = cms.InputTag('pat'+metType+'Met' + postfix),
                       srcCorrections = cms.VInputTag(corrections)
                     )
            sequenceName="patMetCorrectionSequence"
            
        #MM: FIXME MVA
        #if metType == "MVA":
        #    return patMetCorrectionSequence, metModName #FIXME
        #    corMetProducer = self.createMVAMETModule(process)
        #    sequenceName="pfMVAMEtSequence"
            
        setattr(process,metModName, corMetProducer)
        
        # adding the full sequence only if it does not exist
        if not hasattr(process, sequenceName+postfix):
            for corModule in correctionSequence:
                patMetCorrectionSequence += corModule
            setattr(process, sequenceName+postfix, patMetCorrectionSequence)
            
        else: #if it exists, only add the missing correction modules, no need to redo everything
            patMetCorrectionSequence = cms.Sequence()
            setattr(process, sequenceName+postfix,patMetCorrectionSequence)
            for mod in corModNames.keys():
                if not hasattr(process, corModNames[mod]):
                    patMetCorrectionSequence += corModule


        #plug the main patMetproducer
        metModuleSequence += getattr(process, metModName)
        
        #create the intermediate MET steps
        #and finally add the met producers in the sequence for scheduled mode
        if produceIntermediateCorrections:
            interMets = self.addIntermediateMETs(process, metType, correctionLevel, corScheme, corTags,corNames, postfix)
            for met in interMets.keys():
                setattr(process,met, interMets[met] )
                metModuleSequence += getattr(process, met)

        return patMetCorrectionSequence, metModName
                

#====================================================================================================
    def addIntermediateMETs(self, process, metType, correctionLevel, corScheme, corTags, corNames, postfix):
        interMets = {}

        # we don't want to duplicate an exisiting module if we ask for a simple 1-corr scheme
        if len(correctionLevel) == 1:
            return interMets

        #ugly, but it works
        nCor=len(correctionLevel)+1
        ids = [0]*nCor
        for i in range(nCor**nCor):
            tmp=i
            exists=False
            corName=""
            corrections = []
            for j in range(nCor):
                ids[j] = tmp%nCor
                tmp = tmp//nCor
                
                if j != 0 and ids[j-1] < ids[j]:
                    exists=True
                for k in range(0,j):
                    if ids[k] == ids[j] and ids[k]!=0:
                        exists=True
                            
            if exists or sum(ids[j] for j in range(nCor))==0:
                continue
            
            for cor in range(nCor):
                cid = ids[nCor-cor-1]
                cKey = correctionLevel[cid-1]
                if cid ==0:#empty correction
                    continue
                else :
                    corName += corNames[cKey]
                    corrections.append( corTags[ cKey ] )

            if corName == corScheme:
                continue

            corName='pat'+metType+'Met' + corName + postfix
            interMets[corName] =  cms.EDProducer("CorrectedPATMETProducer",
                 src = cms.InputTag('pat'+metType+'Met' + postfix),
                 srcCorrections = cms.VInputTag(corrections)
               )
        

        return interMets

                
#====================================================================================================
    def getMETUncertainties(self, process, metType, metModName, electronCollection,
                            photonCollection, muonCollection, tauCollection, 
                            pfCandCollection, jetCollection, jetUncInfos, 
                            postfix):

        
        # uncertainty sequence
        metUncSequence = cms.Sequence()
        shiftedModuleSequence = cms.Sequence()

        #===================================================================================
        # jet energy resolution shifts
        #===================================================================================
        if not isValidInputTag(jetCollection): #or jetCollection=="":
            print "INFO : jet collection %s does not exists, no energy resolution shifting will be performed in MET uncertainty tools" % jetCollection
        else: 
            preId=""
            if "Smear" in metModName:
                preId="Smeared"

            metJERUncModules = self.getVariations(process, metModName, "Jet",preId, jetCollection, "Res", metUncSequence, postfix=postfix )
            
            for mod in metJERUncModules.keys():
                setattr(process, mod, metJERUncModules[mod] )
                shiftedModuleSequence += getattr(process, mod)

        #===================================================================================
        # Unclustered energy candidates
        #===================================================================================
        #Jet projection ==
        pfCandsNoJets = cms.EDProducer("CandPtrProjector", 
                                          src = pfCandCollection, 
                                          veto = jetCollection
                                          )
        setattr(process, "pfCandsNoJets"+postfix, pfCandsNoJets)
        metUncSequence += getattr(process, "pfCandsNoJets"+postfix)
                
        #electron projection ==
        pfCandsNoJetsNoEle = cms.EDProducer("CandPtrProjector", 
                                          src = cms.InputTag("pfCandsNoJets"+postfix),
                                          veto = electronCollection
                                          )
        setattr(process, "pfCandsNoJetsNoEle"+postfix, pfCandsNoJetsNoEle)
        metUncSequence += getattr(process, "pfCandsNoJetsNoEle"+postfix)
        
        #muon projection ==
        pfCandsNoJetsNoEleNoMu = cms.EDProducer("CandPtrProjector", 
                                          src = cms.InputTag("pfCandsNoJetsNoEle"+postfix),
                                          veto = muonCollection
                                          )
        setattr(process, "pfCandsNoJetsNoEleNoMu"+postfix, pfCandsNoJetsNoEleNoMu)
        metUncSequence += getattr(process, "pfCandsNoJetsNoEleNoMu"+postfix)
        
        #tau projection ==
        pfCandsNoJetsNoEleNoMuNoTau = cms.EDProducer("CandPtrProjector", 
                                          src = cms.InputTag("pfCandsNoJetsNoEleNoMu"+postfix),
                                          veto = tauCollection
                                          )
        setattr(process, "pfCandsNoJetsNoEleNoMuNoTau"+postfix, pfCandsNoJetsNoEleNoMuNoTau)
        metUncSequence += getattr(process, "pfCandsNoJetsNoEleNoMuNoTau"+postfix)
        
        #photon projection ==
        pfCandsForUnclusteredUnc = cms.EDProducer("CandPtrProjector", 
                                          src = cms.InputTag("pfCandsNoJetsNoEleNoMuNoTau"+postfix),
                                          veto = tauCollection
                                          )
        setattr(process, "pfCandsForUnclusteredUnc"+postfix, pfCandsForUnclusteredUnc)
        metUncSequence += getattr(process, "pfCandsForUnclusteredUnc"+postfix)
                
        #===================================================================================
        # energy shifts
        #===================================================================================
        objectCollections = { "Jet":jetCollection,
                              "Electron":electronCollection,
                              "Photon":photonCollection,
                              "Muon":muonCollection,
                              "Unclustered":cms.InputTag("pfCandsForUnclusteredUnc"+postfix),
                              "Tau":tauCollection,
                              }
        
        for obj in objectCollections.keys():
            if not isValidInputTag(objectCollections[obj]): # or objectCollections[obj]=="":
                print "INFO : %s collection %s does not exists, no energy scale shifting will be performed in MET uncertainty tools" %(obj, objectCollections[obj])
            else:
                metObjUncModules = self.getVariations(process, metModName, obj,"", objectCollections[obj], "En", metUncSequence, jetUncInfos, postfix )
                
                #adding the shifted MET produced to the proper patMetModuleSequence
                for mod in metObjUncModules.keys():
                    setattr(process, mod, metObjUncModules[mod] )
                    shiftedModuleSequence += getattr(process, mod)
                    
        #return the sequence containing the shifted collections producers
        return metUncSequence, shiftedModuleSequence

#====================================================================================================
    def createEnergyScaleShiftedUpModule(self, process,identifier, objectCollection,
                                         varyByNsigmas, jetUncInfos=None, postfix=""):

        shiftedModuleUp = None
        
        if identifier == "Electron":
            shiftedModuleUp = cms.EDProducer("ShiftedParticleProducer",
                                             src = objectCollection,
                                             uncertainty = cms.string('((abs(y)<1.479)?(0.006+0*x):(0.015+0*x))'),
                                             shiftBy = cms.double(+1.*varyByNsigmas)
                                             )
            
        if identifier == "Photon":
            shiftedModuleUp = cms.EDProducer("ShiftedParticleProducer",
                                             src = objectCollection,
                                             uncertainty = cms.string('((abs(y)<1.479)?(0.01+0*x):(0.025+0*x))'),
                                             shiftBy = cms.double(+1.*varyByNsigmas)
                                             )

        if identifier == "Muon":
            shiftedModuleUp = cms.EDProducer("ShiftedParticleProducer",
                                             src = objectCollection,
                                             uncertainty = cms.string('((x<100)?(0.002+0*y):(0.05+0*y))'),
                                             shiftBy = cms.double(+1.*varyByNsigmas)
                                             )
            
        if identifier == "Tau":
            shiftedModuleUp = cms.EDProducer("ShiftedParticleProducer",
                                             src = objectCollection,
                                             uncertainty = cms.string('0.03+0*x*y'),
                                             shiftBy = cms.double(+1.*varyByNsigmas)
                                             )

        if identifier == "Unclustered":        
            shiftedModuleUp = cms.EDProducer("ShiftedParticleProducer",
                                             src = objectCollection,
                                             binning = cms.VPSet(
                    # charged PF hadrons - tracker resolution
                    cms.PSet(
                        binSelection = cms.string('charge!=0'),
                        binUncertainty = cms.string('sqrt(pow(0.00009*x,2)+pow(0.0085/sqrt(sin(2*atan(exp(-y)))),2))')
                        ),
                    # neutral PF hadrons - HCAL resolution
                    cms.PSet(
                        binSelection = cms.string('pdgId==130'),
                        energyDependency = cms.bool(True),
                        binUncertainty = cms.string('((abs(y)<1.3)?(min(0.25,sqrt(0.64/x+0.0025))):(min(0.30,sqrt(1.0/x+0.0016))))')
                        ),
                    # photon - ECAL resolution
                    cms.PSet(
                        binSelection = cms.string('pdgId==22'),
                        energyDependency = cms.bool(True),
                        binUncertainty = cms.string('sqrt(0.0009/x+0.000001)+0*y')
                        ),
                    # HF particules - HF resolution
                    cms.PSet(
                        binSelection = cms.string('pdgId==1 || pdgId==2'),
                        energyDependency = cms.bool(True),
                        binUncertainty = cms.string('sqrt(1./x+0.0025)+0*y')
                        ),
                    ),
                                             shiftBy = cms.double(+1.*varyByNsigmas)
                                             )

        if identifier == "Jet":
            moduleType="ShiftedPATJetProducer"
            #MM: FIXME MVA
            #if self._parameters["metType"].value == "MVA":
            #    moduleType="ShiftedPFJetProducer"

            if jetUncInfos["jecUncFile"] == "":
                shiftedModuleUp = cms.EDProducer(moduleType,
                                                 src = objectCollection,
                                                 jetCorrUncertaintyTag = cms.string(jetUncInfos["jecUncTag"] ),
                                                 addResidualJES = cms.bool(True),
                                                 jetCorrLabelUpToL3 = cms.InputTag(jetUncInfos["jCorLabelUpToL3"] ),
                                                 jetCorrLabelUpToL3Res = cms.InputTag(jetUncInfos["jCorLabelL3Res"] ),
                                                 jetCorrPayloadName =  cms.string(jetUncInfos["jCorrPayload"] ),
                                                 shiftBy = cms.double(+1.*varyByNsigmas),
                                                 )
            else:
                shiftedModuleUp = cms.EDProducer(moduleType,
                                                 src = objectCollection,
                                                 jetCorrInputFileName = cms.FileInPath(jetUncInfos["jecUncFile"] ),
                                                 jetCorrUncertaintyTag = cms.string(jetUncInfos["jecUncTag"] ),
                                                 addResidualJES = cms.bool(True),
                                                 jetCorrLabelUpToL3 = cms.InputTag(jetUncInfos["jCorLabelUpToL3"] ),
                                                 jetCorrLabelUpToL3Res = cms.InputTag(jetUncInfos["jCorLabelL3Res"] ),
                                                 jetCorrPayloadName =  cms.string(jetUncInfos["jCorrPayload"] ),
                                                 shiftBy = cms.double(+1.*varyByNsigmas),
                                                 )


        return shiftedModuleUp


#====================================================================================================


#====================================================================================================
    def removePostfix(self, name, postfix):
        
        if postfix=="":
            return name

        baseName = name
        if baseName[-len(postfix):] == postfix:
            baseName = baseName[0:-len(postfix)]
        else:
            raise Exception("Tried to remove postfix %s from %s, but it wasn't there" % (postfix, baseName))
        
        return baseName

#====================================================================================================
    def tuneTxyParameters(self, process, corScheme, postfix):
        import PhysicsTools.PatUtils.patPFMETCorrections_cff as metCors
        xyTags = {
            "Txy_50ns":metCors.patMultPhiCorrParams_Txy_50ns,
            "T1Txy_50ns":metCors.patMultPhiCorrParams_T1Txy_50ns,
            "T0pcTxy_50ns":metCors.patMultPhiCorrParams_T0pcTxy_50ns,
            "T0pcT1Txy_50ns":metCors.patMultPhiCorrParams_T0pcT1Txy_50ns,
            "T1T2Txy_50ns":metCors.patMultPhiCorrParams_T1T2Txy_50ns,
            "T0pcT1T2Txy_50ns":metCors.patMultPhiCorrParams_T0pcT1T2Txy_50ns,
            "T1SmearTxy_50ns":metCors.patMultPhiCorrParams_T1SmearTxy_50ns,
            "T1T2SmearTxy_50ns":metCors.patMultPhiCorrParams_T1T2SmearTxy_50ns,
            "T0pcT1SmearTxy_50ns":metCors.patMultPhiCorrParams_T0pcT1SmearTxy_50ns,
            "T0pcT1T2SmearTxy_50ns":metCors.patMultPhiCorrParams_T0pcT1T2SmearTxy_50ns,

            "Txy_25ns":metCors.patMultPhiCorrParams_Txy_25ns,
            "T1Txy_25ns":metCors.patMultPhiCorrParams_T1Txy_25ns,
            "T0pcTxy_25ns":metCors.patMultPhiCorrParams_T0pcTxy_25ns,
            "T0pcT1Txy_25ns":metCors.patMultPhiCorrParams_T0pcT1Txy_25ns,
            "T1T2Txy_25ns":metCors.patMultPhiCorrParams_T1T2Txy_25ns,
            "T0pcT1T2Txy_25ns":metCors.patMultPhiCorrParams_T0pcT1T2Txy_25ns,
            "T1SmearTxy_25ns":metCors.patMultPhiCorrParams_T1SmearTxy_25ns,
            "T1T2SmearTxy_25ns":metCors.patMultPhiCorrParams_T1T2SmearTxy_25ns,
            "T0pcT1SmearTxy_25ns":metCors.patMultPhiCorrParams_T0pcT1SmearTxy_25ns,
            "T0pcT1T2SmearTxy_25ns":metCors.patMultPhiCorrParams_T0pcT1T2SmearTxy_25ns
            }
        
        getattr(process, "patPFMetTxyCorr"+postfix).parameters = xyTags[corScheme+"_25ns"] 
        ##for automatic switch to 50ns / 25ns corrections ==> does not work...
        #from Configuration.StandardSequences.Eras import eras
        #eras.run2_50ns_specific.toModify( getattr(process, "patPFMetTxyCorr"+postfix) , parameters=xyTags[corScheme+"_50ns"] )
        #eras.run2_25ns_specific.toModify( getattr(process, "patPFMetTxyCorr"+postfix) , parameters=xyTags[corScheme+"_25ns"] )


#====================================================================================================
    def getVariations(self, process, metModName, identifier,preId, objectCollection, varType, 
                      metUncSequence, jetUncInfos=None, postfix="" ):

        # temporary hardcoded varyByNSigma value
        varyByNsigmas=1

        # remove the postfix to put it at the end
        baseName = self.removePostfix(metModName, postfix)

        #default shifted MET producers
        shiftedMetProducers = {preId+identifier+varType+'Up':None, preId+identifier+varType+'Down':None}

        #create the shifted collection producers=========================================
        shiftedCollModules = {'Up':None, 'Down':None}
        
        if identifier=="Jet" and varType=="Res":
            smear=False
            if "Smear" in metModName:
                smear=True
     
            shiftedCollModules['Up'] = self.createShiftedJetResModule(process, smear, objectCollection, +1.*varyByNsigmas,
                                                                 "Up", metUncSequence, postfix)
            shiftedCollModules['Down'] = self.createShiftedJetResModule(process, smear, objectCollection, -1.*varyByNsigmas,
                                                                   "Down", metUncSequence, postfix)
       
        else:
            shiftedCollModules['Up'] = self.createEnergyScaleShiftedUpModule(process, identifier, objectCollection, varyByNsigmas, jetUncInfos, postfix)
            shiftedCollModules['Down'] = shiftedCollModules['Up'].clone( shiftBy = cms.double(-1.*varyByNsigmas) )

        if identifier=="Jet" and varType=="Res":
            smear=False
            if "Smear" in metModName:
                objectCollection=cms.InputTag("selectedPatJetsForMetT1T2SmearCorr"+postfix)



        #and the MET producers
        shiftedMetProducers = self.createShiftedModules(process, shiftedCollModules, identifier, preId, objectCollection, 
                                                        metModName, varType, metUncSequence, postfix)

        return shiftedMetProducers

#========================================================================================
    def createShiftedJetResModule(self, process, smear, objectCollection, varyByNsigmas, varDir, metUncSequence, postfix ):
        
        smearedJetModule = self.createSmearedJetModule(process, objectCollection, smear, varyByNsigmas, varDir, metUncSequence, postfix)

        return smearedJetModule


#========================================================================================
    def createShiftedModules(self, process, shiftedCollModules, identifier, preId, objectCollection, metModName, varType, metUncSequence, postfix):

        shiftedMetProducers = {}

        # remove the postfix to put it at the end
        baseName = self.removePostfix(metModName, postfix)
       
        #adding the shifted collection producers to the sequence, create the shifted MET correction Modules and add them as well
        for mod in shiftedCollModules.keys():
            modName = "shiftedPat"+preId+identifier+varType+mod+postfix
            #MM: FIXME MVA
            #if  "MVA" in metModName and identifier == "Jet": #dummy fix
            #    modName = "uncorrectedshiftedPat"+preId+identifier+varType+mod+postfix
            setattr(process, modName, shiftedCollModules[mod])
            metUncSequence += getattr(process, modName)
            
            #removing the uncorrected
            modName = "shiftedPat"+preId+identifier+varType+mod+postfix

            #PF MET =================================================================================
            if "PF" in metModName:
                #create the MET shifts and add them to the sequence
                shiftedMETCorrModule = self.createShiftedMETModule(process, objectCollection, modName)
                modMETShiftName = "shiftedPatMETCorr"+preId+identifier+varType+mod+postfix
                setattr(process, modMETShiftName, shiftedMETCorrModule)
                metUncSequence += getattr(process, modMETShiftName)
                
                #and finally prepare the shifted MET producers
                modName = baseName+identifier+varType+mod+postfix
                shiftedMETModule = getattr(process, metModName).clone(
                    src = cms.InputTag( metModName ),
                    srcCorrections = cms.VInputTag( cms.InputTag(modMETShiftName) )
                    )
                shiftedMetProducers[ modName ] = shiftedMETModule

            #MM: FIXME MVA
            #MVA MET, duplication of the MVA MET producer ============================================
            #if "MVA" in metModName:
            #    print "name: ",metModName, modName 
            #    shiftedMETModule = self.createMVAMETModule(process, identifier, modName, True)
            #    modName = baseName+identifier+varType+mod+postfix
            #    setattr(process, modName, shiftedMETModule)
            #    shiftedMetProducers[ modName ] = shiftedMETModule
            #    
            #     #pileupjetId and  =====
            #    if identifier == "Jet":
            #        #special collection replacement for the MVAMET for the jet case ======
            #        origCollection = cms.InputTag("calibratedAK4PFJetsForPFMVAMEt"+postfix) #self._parameters["jetCollection"].value
            #        newCollection = cms.InputTag("uncorrectedshiftedPat"+preId+identifier+varType+mod+postfix)
            #        moduleName = "shiftedPat"+preId+identifier+varType+mod+postfix
            #        corrShiftedModule = getattr(process,"calibratedAK4PFJetsForPFMVAMEt").clone(
            #            src=newCollection
            # )

            #        setattr(process, moduleName, corrShiftedModule)
            #        metUncSequence += getattr(process, moduleName)

            #        puJetIdProducer = getattr(process, "puJetIdForPFMVAMEt").clone(
            #            jets = moduleName
            #            )
            #        puJetIdName = "puJetIdForPFMVAMEt"+preId+identifier+varType+mod+postfix
            #        setattr(process, puJetIdName, puJetIdProducer)
            #        metUncSequence += getattr(process, puJetIdName)
            #        shiftedMETModule.srcMVAPileupJetId = cms.InputTag(puJetIdName,"fullDiscriminant")
             
           #==========================================================================================

        return shiftedMetProducers


#========================================================================================
    def createShiftedMETModule(self, process, originCollection, shiftedCollection):

        shiftedModule = cms.EDProducer("ShiftedParticleMETcorrInputProducer",
                                       srcOriginal = originCollection,
                                       srcShifted = cms.InputTag(shiftedCollection),
                                       )

        return shiftedModule

#========================================================================================
    def createMVAMETModule(self, process, identifier="", shiftedCollection="", isShifted=False, postfix="" ):

        if not hasattr(process, "pfMVAMEt"):
            process.load("RecoMET.METPUSubtraction.mvaPFMET_cff")
 
        #retrieve collections
        electronCollection = self._parameters["electronCollection"].value
        muonCollection = self._parameters["electronCollection"].value
        photonCollection = self._parameters["photonCollection"].value
        tauCollection = self._parameters["tauCollection"].value
        pfCandCollection = self._parameters["pfCandCollection"].value
        corJetCollection = cms.InputTag("calibratedAK4PFJetsForPFMVAMEt"+postfix)
        uncorJetCollection = cms.InputTag("ak4PFJets")

        #shift if needed===
        if isShifted:
            if identifier == "Electron":
                electronCollection = cms.InputTag(shiftedCollection)
            if identifier == "Muon":
                muonCollection = cms.InputTag(shiftedCollection)
            if identifier == "Tau":
                tauCollection = cms.InputTag(shiftedCollection)
            if identifier == "Photon":
                photonCollection = cms.InputTag(shiftedCollection)
            if identifier == "Unclustered":
                pfCandCollection = cms.InputTag(shiftedCollection)
            if identifier == "Jet":
                corJetCollection = cms.InputTag(shiftedCollection)
                uncorJetCollection = cms.InputTag("uncorrected"+shiftedCollection)
                

        #leptons
        mvaMetLeptons = self._parameters["mvaMetLeptons"].value
        leptons = cms.VInputTag([])
        if "Electrons" in mvaMetLeptons and isValidInputTag(electronCollection):
            leptons.append = electronCollection
        if "Muons" in mvaMetLeptons and isValidInputTag(muonCollection):
            leptons.append = muonCollection
        if "Photons" in mvaMetLeptons and isValidInputTag(photonCollection):
            leptons.append = photonCollection
        if "Taus" in mvaMetLeptons and isValidInputTag(tauCollection):
            leptons.append = tauCollection


        mvaMetProducer=getattr(process, "pfMVAMEt").clone( 
            srcCorrJets = corJetCollection,
            srcUncorrJets = uncorJetCollection,
            srcPFCandidates = pfCandCollection,
            srcLeptons = leptons,
            )
        
        return mvaMetProducer

#========================================================================================
    def getUnclusteredVariationsForMVAMET(self, process, var, val,  metUncSequence, postfix ):

        if not hasattr(process, "pfCandsNotInJetsForMetCorr"):
            process.load("JetMETCorrections.Type1MET.correctionTerms.PfMetType1Type2_cff")
 
        #MM: it's bloody stupid to make it that way....
        # compute the shifted particles ====
        unclCandModule = cms.EDProducer("ShiftedPFCandidateProducer",
                                        src = cms.InputTag('pfCandsNotInJetsForMetCorr'),
                                        shiftBy = cms.double(val),
                                        uncertainty = cms.double(0.10)
                                        )
        setattr(process, "pfCandsNotInJetsUnclusteredEn"+var+postfix, unclCandModule)
        metUncSequence += getattr(process, "pfCandsNotInJetsUnclusteredEn"+var+postfix)
            
     

        #replace the old unclustered particles by the shifted ones....
        pfCandCollection = self._parameters["pfCandCollection"].value

        #top projection on jets
        pfCandsNotInJets = cms.EDProducer("CandPtrProjector", 
                                          src = pfCandCollection, 
                                          veto = cms.InputTag("ak4PFJets")
                                          )
        setattr(process, "pfCandsNotInJetsUnclusteredEn"+var+postfix, pfCandsNotInJets)
        metUncSequence += getattr(process,"pfCandsNotInJetsUnclusteredEn"+var+postfix)

        fullShiftedModule = self.createShiftedObjectModuleForMVAMET(pfCandCollection, cms.InputTag("pfCandsNotInJetsUnclusteredEn"+var+postfix), 0.01 )
        setattr(process, "pfCandidatesEn"+var+postfix, fullShiftedModule)
        metUncSequence += getattr(process, "pfCandidatesEn"+var+postfix)

        # duplication of the MVA MET producer ============================================
        shiftedMETModule = self.createMVAMETModule(process, "Unclustered", "pfCandidatesEn"+var+postfix, True)
        return shiftedMETModule

#========================================================================================
    def createShiftedObjectModuleForMVAMET(self, origCollection, shiftedCollection, dr=0.5):
        fullShiftedModule = cms.EDProducer("ShiftedPFCandidateProducerByMatchedObject",
                 srcPFCandidates = origCollection,
                 srcUnshiftedObjects = origCollection,
                 dRmatch_PFCandidate = cms.double(dr),
                 srcShiftedObjects = shiftedCollection
               )
        return fullShiftedModule

#========================================================================================
    def createSmearedJetModule(self, process, jetCollection, smear, varyByNsigmas, varDir, metUncSequence, postfix):
        
        smearedJetModule = None

        modName = "pat"
        selJetModName= "selectedPatJetsForMetT1T2"
        if smear:
            modName += "SmearedJets"
            selJetModName += "SmearCorr"
        else:
            modName += "Jets"
            
            
        if varDir != "":
            modName += "Res"+varDir
            selJetModName += "Res"+varDir

        modName += postfix
        selJetModName += postfix

        if "PF" == self._parameters["metType"].value:
            smearedJetModule = getattr(process, "patSmearedJets"+postfix).clone(
                src = cms.InputTag(jetCollection.value()+postfix),
                enabled = cms.bool(smear),
                variation = cms.int32( int(varyByNsigmas) ),
                )    
           
        #MM: FIXME MVA
        #if "MVA" == self._parameters["metType"].value:
        #    from RecoMET.METProducers.METSigParams_cfi import *

        #    genJetsCollection=cms.InputTag('ak4GenJetsNoNu')
        #    if self._parameters["onMiniAOD"].value:
        #        genJetsCollection=cms.InputTag("slimmedGenJets")

        #    smearedJetModule = cms.EDProducer("SmearedPFJetProducer",
        #            src = cms.InputTag('ak4PFJets'),
        #            jetCorrLabel = cms.InputTag("ak4PFL1FastL2L3Corrector"),
        #            dRmaxGenJetMatch = cms.string('min(0.5, 0.1 + 0.3*exp(-0.05*(genJetPt - 10.)))'),
        #            sigmaMaxGenJetMatch = cms.double(3.),
        #            inputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/pfJetResolutionMCtoDataCorrLUT.root'),
        #            lutName = cms.string('pfJetResolutionMCtoDataCorrLUT'),
        #            jetResolutions = METSignificance_params,
        #            skipRawJetPtThreshold = cms.double(10.), # GeV
        #            skipCorrJetPtThreshold = cms.double(1.e-2),
        #            srcGenJets = genJetsCollection,
        #            shiftBy = cms.double(varyByNsigmas),
        #            #verbosity = cms.int32(1)
        #            )

        return smearedJetModule


### Utilities ====================================================================
    def initializeInputTag(self, input, default):
        retVal = None
        if input is None:
            retVal = self._defaultParameters[default].value
        elif type(input) == str:
            retVal = cms.InputTag(input)
        else:
            retVal = input
        return retVal


    def recomputeRawMetFromPfcs(self, process, pfCandCollection, onMiniAOD, patMetModuleSequence,  postfix):
        
        #RECO MET
        if not hasattr(process, "pfMet"+postfix) and self._parameters["metType"].value == "PF":
            #common to AOD/mAOD processing
            #raw MET
            from RecoMET.METProducers.PFMET_cfi import pfMet
            setattr(process, "pfMet"+postfix, pfMet.clone() )
            getattr(process, "pfMet"+postfix).src = pfCandCollection
            getattr(process, "pfMet"+postfix).calculateSignificance = False
            patMetModuleSequence += getattr(process, "pfMet"+postfix)
            
            #PAT METs
            process.load("PhysicsTools.PatAlgos.producersLayer1.metProducer_cff")
            configtools.cloneProcessingSnippet(process, getattr(process,"patMETCorrections"), postfix)
                  
            #T1 pfMet for AOD to mAOD only
            if not onMiniAOD:
                #correction duplication needed
                getattr(process, "pfMetT1"+postfix).src = cms.InputTag("pfMet"+postfix)
                patMetModuleSequence += getattr(process, "pfMetT1"+postfix)

                setattr(process, 'patMETs'+postfix, getattr(process,'patMETs' ).clone() )
                getattr(process, "patMETs"+postfix).metSource = cms.InputTag("pfMetT1"+postfix)


    def extractMET(self, process, correctionLevel, patMetModuleSequence, postfix):
        pfMet = cms.EDProducer("RecoMETExtractor",
                               metSource= cms.InputTag("slimmedMETs",processName=cms.InputTag.skipCurrentProcess()),
                               correctionLevel = cms.string(correctionLevel)
                               )
        if(correctionLevel=="raw"):
            setattr(process,"pfMet"+postfix ,pfMet)
            patMetModuleSequence += getattr(process, "pfMet"+postfix)
        else:
            setattr(process,"met"+correctionLevel+postfix ,pfMet)
            patMetModuleSequence += getattr(process, "met"+correctionLevel+postfix)
   
        genMetExtractor = cms.EDProducer("GenMETExtractor",
                                             metSource= cms.InputTag("slimmedMETs",processName=cms.InputTag.skipCurrentProcess())
                                             )
        setattr(process, "genMetExtractor"+postfix ,genMetExtractor)
        #patMetModuleSequence += getattr(process, "genMetExtractor"+postfix)
                                             
        
    def updateJECs(self,process,jetCollection, patMetModuleSequence, postfix):
        from PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cff import updatedPatJetCorrFactors
        
        patJetCorrFactorsReapplyJEC = updatedPatJetCorrFactors.clone(
            src = cms.InputTag("slimmedJets"),
            levels = ['L1FastJet', 
                      'L2Relative', 
                      'L3Absolute'],
            payload = 'AK4PFchs' ) # always CHS from miniAODs
        
        if self._parameters["runOnData"].value:
            patJetCorrFactorsReapplyJEC.levels.append("L2L3Residual")

        from PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cff import updatedPatJets
        patJetsReapplyJEC = updatedPatJets.clone(
            jetSource = cms.InputTag("slimmedJets"),
            jetCorrFactorsSource = cms.VInputTag(cms.InputTag("patJetCorrFactorsReapplyJEC"+postfix))
            )
        
        setattr(process,"patJetCorrFactorsReapplyJEC"+postfix,patJetCorrFactorsReapplyJEC)
        setattr(process,"patJetsReapplyJEC"+postfix,patJetsReapplyJEC.clone())
        patMetModuleSequence += getattr(process,"patJetCorrFactorsReapplyJEC"+postfix)
        patMetModuleSequence += getattr(process,"patJetsReapplyJEC"+postfix)
        
        return  cms.InputTag("patJetsReapplyJEC"+postfix)


    def getJetCollectionForCorsAndUncs(self, process, jetCollectionUnskimmed, 
                                       jetSelection, autoJetCleaning,patMetModuleSequence, postfix):

        basicJetsForMet = cms.EDProducer("PATJetCleanerForType1MET",
                                         src = jetCollectionUnskimmed,
                                         jetCorrEtaMax = cms.double(9.9),
                                         jetCorrLabel = cms.InputTag("L3Absolute"),
                                         jetCorrLabelRes = cms.InputTag("L2L3Residual"),
                                         offsetCorrLabel = cms.InputTag("L1FastJet"),
                                         skipEM = cms.bool(True),
                                         skipEMfractionThreshold = cms.double(0.9),
                                         skipMuonSelection = cms.string('isGlobalMuon | isStandAloneMuon'),
                                         skipMuons = cms.bool(True),
                                         type1JetPtThreshold = cms.double(15.0)
                                         )
        setattr(process, "basicJetsForMet"+postfix, basicJetsForMet)
        patMetModuleSequence += getattr(process, "basicJetsForMet"+postfix)
        
        from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets
        jetSelector = selectedPatJets.clone(
            src = cms.InputTag("basicJetsForMet"+postfix),
            cut = cms.string(jetSelection)
            )
        setattr(process, "jetSelectorForMet"+postfix, jetSelector)
        patMetModuleSequence += getattr(process, "jetSelectorForMet"+postfix)
        
        jetCollection = self.jetCleaning(process, "jetSelectorForMet"+postfix, autoJetCleaning, patMetModuleSequence, postfix)

        return jetCollection


    def ak4JetReclustering(self,process, pfCandCollection, patMetModuleSequence, postfix):
        
        chs = self._parameters["CHS"].value
        jetColName="ak4PFJets"
        CHSname=""
        pfCandColl=pfCandCollection
        if chs:
            CHSname="chs"
            jetColName="ak4PFJetsCHS"

            pfCHS=None
            if self._parameters["onMiniAOD"].value: 
                pfCHS = cms.EDFilter("CandPtrSelector", src = pfCandCollection, cut = cms.string("fromPV"))
            else:
                setattr( process, "tmpPFCandCollPtr", cms.EDProducer("PFCandidateFwdPtrProducer",
                                                                     src = pfCandCollection ) )
                process.load("CommonTools.ParticleFlow.pfNoPileUpJME_cff")
                configtools.cloneProcessingSnippet(process, getattr(process,"pfNoPileUpJMESequence"), postfix )
                getattr(process, "pfPileUpJME"+postfix).PFCandidates = cms.InputTag("tmpPFCandCollPtr")
                pfCHS = getattr(process, "pfNoPileUpJME").clone( bottomCollection = cms.InputTag("tmpPFCandCollPtr") )
            
            if not hasattr(process, "pfCHS"+postfix):
                setattr(process,"pfCHS"+postfix,pfCHS)
            pfCandColl = cms.InputTag("pfCHS"+postfix)
            

        jetColName+=postfix
        if not hasattr(process, jetColName):
            process.load("RecoJets.JetProducers.ak4PFJets_cfi")
            
            #if chs:
            setattr(process, jetColName, getattr(process,"ak4PFJets").clone() )

            getattr(process, jetColName).src = pfCandColl 
            getattr(process, jetColName).doAreaFastjet = True
            
            patMetModuleSequence += getattr(process, jetColName)

            corLevels=['L1FastJet', 'L2Relative', 'L3Absolute']
            if self._parameters["runOnData"].value:
                corLevels.append("L2L3Residual")

            switchJetCollection(process,
                                jetSource = cms.InputTag(jetColName),
                                jetCorrections = ('AK4PF'+CHSname, corLevels , ''),
                                postfix=postfix
                                )

            getattr(process,"patJets"+postfix).addGenJetMatch = False 
            getattr(process,"patJets"+postfix).addGenPartonMatch = False 
            getattr(process,"patJets"+postfix).addPartonJetMatch = False 
            if self._parameters['onMiniAOD'].value:
                del getattr(process,"patJets"+postfix).JetFlavourInfoSource
                del getattr(process,"patJets"+postfix).JetPartonMapSource
            getattr(process,"patJets"+postfix).getJetMCFlavour = False
            
            getattr(process,"patJetCorrFactors"+postfix).src=cms.InputTag(jetColName)
            getattr(process,"patJetCorrFactors"+postfix).primaryVertices= cms.InputTag("offlineSlimmedPrimaryVertices")

         
        return cms.InputTag("patJets"+postfix)
        

    def miniAODConfiguration(self, process, pfCandCollection, jetCollection,
                             patMetModuleSequence, postfix ):      
        if self._parameters["metType"].value == "PF": # not hasattr(process, "pfMet"+postfix)
            if "T1" in self._parameters['correctionLevel'].value:
                getattr(process, "patPFMet"+postfix).srcJets = cms.InputTag(jetCollection.value()+postfix)
                getattr(process, "patPFMet"+postfix).srcLeptons = cms.VInputTag(self._parameters["electronCollection"].value, 
                                                                                self._parameters["muonCollection"].value,
                                                                                self._parameters["photonCollection"].value,
                                                                                )
            getattr(process, "patPFMet"+postfix).addGenMET  = False
            if not self._parameters["runOnData"].value:
                getattr(process, "patPFMet"+postfix).addGenMET  = True
                getattr(process, "patPFMet"+postfix).genMETSource = cms.InputTag("genMetExtractor"+postfix)

            if hasattr(process, "patPFMetTxyCorr"+postfix):
                getattr(process, "patPFMetTxyCorr"+postfix).vertexCollection = cms.InputTag("offlineSlimmedPrimaryVertices")


        if "Smear" in self._parameters['correctionLevel'].value:
            getattr(process, "patSmearedJets").genJets = cms.InputTag("slimmedGenJets")

        #MM: FIXME MVA
        #if hasattr(process, "pfMVAMet"):
        #    getattr(process, "pfMVAMet").srcVertices = cms.InputTag("offlineSlimmedPrimaryVertices")
        #    getattr(process, "pfMVAMEt").srcVertices = cms.InputTag("offlineSlimmedPrimaryVertices")
        #    getattr(process, "puJetIdForPFMVAMEt").vertexes = cms.InputTag("offlineSlimmedPrimaryVertices")
        #    getattr(process, "patMVAMet").addGenMET  = False

        if not hasattr(process, "slimmedMETs"+postfix) and self._parameters["metType"].value == "PF":

            from PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi import slimmedMETs
            setattr(process, "slimmedMETs"+postfix, slimmedMETs.clone() )
            getattr(process,"slimmedMETs"+postfix).src = cms.InputTag("patPFMetT1"+postfix)
            getattr(process,"slimmedMETs"+postfix).rawVariation = cms.InputTag("patPFMet"+postfix)
            getattr(process,"slimmedMETs"+postfix).t1Uncertainties = cms.InputTag("patPFMetT1%s"+postfix)
            getattr(process,"slimmedMETs"+postfix).t01Variation = cms.InputTag("patPFMetT0pcT1"+postfix)
            getattr(process,"slimmedMETs"+postfix).t1SmearedVarsAndUncs = cms.InputTag("patPFMetT1Smear%s"+postfix)
            
            getattr(process,"slimmedMETs"+postfix).tXYUncForRaw = cms.InputTag("patPFMetTxy"+postfix)
            getattr(process,"slimmedMETs"+postfix).tXYUncForT1 = cms.InputTag("patPFMetT1Txy"+postfix)
            getattr(process,"slimmedMETs"+postfix).tXYUncForT01 = cms.InputTag("patPFMetT0pcT1Txy"+postfix)
            getattr(process,"slimmedMETs"+postfix).tXYUncForT1Smear = cms.InputTag("patPFMetT1SmearTxy"+postfix)
            getattr(process,"slimmedMETs"+postfix).tXYUncForT01Smear = cms.InputTag("patPFMetT0pcT1SmearTxy"+postfix)

            getattr(process,"slimmedMETs"+postfix).runningOnMiniAOD = True
            getattr(process,"slimmedMETs"+postfix).t01Variation = cms.InputTag("slimmedMETs",processName=cms.InputTag.skipCurrentProcess())
         
            #extractor for caloMET === temporary for the beginning of the data taking
            self.extractMET(process,"rawCalo",patMetModuleSequence,postfix)
            from PhysicsTools.PatAlgos.tools.metTools import addMETCollection
            addMETCollection(process,
                             labelName = "patCaloMet",
                             metSource = "metrawCalo")
            getattr(process,"patCaloMet").addGenMET = False

            #smearing and type0 variations not yet supported in reprocessing
            #del getattr(process,"slimmedMETs"+postfix).t1SmearedVarsAndUncs
            del getattr(process,"slimmedMETs"+postfix).tXYUncForT01
            del getattr(process,"slimmedMETs"+postfix).tXYUncForT1Smear
            del getattr(process,"slimmedMETs"+postfix).tXYUncForT01Smear
            #del getattr(process,"slimmedMETs"+postfix).caloMET


    def jetConfiguration(self):

        jetFlavor = self._parameters["jetFlavor"].value
        jetCorr = self._parameters["jetCorrectionType"].value

        jetCorLabelUpToL3Name="ak4PF"
        jetCorLabelL3ResName="ak4PF"

        # normal or CHS jets =============================
        if "chs" in jetFlavor:
            self.setParameter("CHS",True)
            jetCorLabelUpToL3Name += "CHS" #chs
            jetCorLabelL3ResName += "CHS"
        elif "Puppi" in jetFlavor:
            self.setParameter("CHS",False)
            
        else:
            self.setParameter("CHS",False)

        # change the correction type =====================
        if jetCorr == "L1L2L3-L1":
            jetCorLabelUpToL3Name += "L1FastL2L3Corrector"
            jetCorLabelL3ResName  += "L1FastL2L3ResidualCorrector"
        elif jetCorr == "L1L2L3-RC": #to be fixed
            jetCorLabelUpToL3Name += "L1FastL2L3Corrector"
            jetCorLabelL3ResName  += "L1FastL2L3ResidualCorrector"

        self.setParameter("jetCorLabelUpToL3",jetCorLabelUpToL3Name )
        self.setParameter("jetCorLabelL3Res",jetCorLabelL3ResName )
        
    # function enabling the auto jet cleaning for uncertainties ===============
    def jetCleaning(self, process, jetCollectionName, autoJetCleaning, jetProductionSequence, postfix ):

        if autoJetCleaning == "None" or autoJetCleaning == "Manual" :
            return cms.InputTag(jetCollectionName)

        #retrieve collections
        electronCollection = self._parameters["electronCollection"].value
        muonCollection = self._parameters["muonCollection"].value
        photonCollection = self._parameters["photonCollection"].value
        tauCollection = self._parameters["tauCollection"].value
        

        if autoJetCleaning == "Full" : # auto clean taus, photons and jets
            if isValidInputTag(tauCollection): 
                process.load("PhysicsTools.PatAlgos.cleaningLayer1.tauCleaner_cfi")
                cleanPatTauProducer = getattr(process, "cleanPatTaus").clone( 
                    src = tauCollection
                    )
                cleanPatTauProducer.checkOverlaps.electrons.src = electronCollection
                cleanPatTauProducer.checkOverlaps.muons.src = muonCollection
                setattr(process, "cleanedPatTaus"+postfix, cleanPatTauProducer)
                jetProductionSequence += getattr(process, "cleanedPatTaus"+postfix)
                tauCollection = cms.InputTag("cleanedPatTaus"+postfix)
            
            if isValidInputTag(photonCollection): 
                process.load("PhysicsTools.PatAlgos.cleaningLayer1.photonCleaner_cfi")
                cleanPatPhotonProducer = getattr(process, "cleanPatPhotons").clone( 
                    src = photonCollection
                    )
                cleanPatPhotonProducer.checkOverlaps.electrons.src = electronCollection
                setattr(process, "cleanedPatPhotons"+postfix, cleanPatPhotonProducer)
                jetProductionSequence += getattr(process, "cleanedPatPhotons"+postfix)
                photonCollection = cms.InputTag("cleanedPatPhotons"+postfix)

        #jet cleaning
        have_cleanPatJets = hasattr(process, "cleanPatJets")
        process.load("PhysicsTools.PatAlgos.cleaningLayer1.jetCleaner_cfi")
        cleanPatJetProducer = getattr(process, "cleanPatJets").clone( 
                     src = cms.InputTag(jetCollectionName)
            )
        #do not leave it hanging
        if not have_cleanPatJets:
            del process.cleanPatJets
        cleanPatJetProducer.checkOverlaps.muons.src = muonCollection
        cleanPatJetProducer.checkOverlaps.electrons.src = electronCollection
        if isValidInputTag(photonCollection) and autoJetCleaning != "LepClean":
            cleanPatJetProducer.checkOverlaps.photons.src = photonCollection
        else:
            del cleanPatJetProducer.checkOverlaps.photons
            
        if isValidInputTag(tauCollection) and autoJetCleaning != "LepClean":
            cleanPatJetProducer.checkOverlaps.taus.src = tauCollection
        else:
            del cleanPatJetProducer.checkOverlaps.taus
           
        # not used at all and electrons are already cleaned
        del cleanPatJetProducer.checkOverlaps.tkIsoElectrons

        setattr(process, "cleanedPatJets"+postfix, cleanPatJetProducer)
        jetProductionSequence += getattr(process, "cleanedPatJets"+postfix)
        return cms.InputTag("cleanedPatJets")


#========================================================================================
runMETCorrectionsAndUncertainties = RunMETCorrectionsAndUncertainties()



#predefined functions for miniAOD production and reproduction
# miniAOD production ===========================
def runMetCorAndUncForMiniAODProduction(process, metType="PF",
                                        jetCollUnskimmed="patJets",
                                        photonColl="selectedPatPhotons",
                                        electronColl="selectedPatElectrons",
                                        muonColl="selectedPatMuons",
                                        tauColl="selectedPatTaus",
                                        pfCandColl = "particleFlow",
                                        jetCleaning="LepClean",
                                        jetSelection="pt>15 && abs(eta)<9.9",
                                        jecUnFile="",
                                        jetFlavor="AK4PFchs",
                                        recoMetFromPFCs=False,
                                        postfix=""):

    runMETCorrectionsAndUncertainties = RunMETCorrectionsAndUncertainties()
    
    #MET flavors
    runMETCorrectionsAndUncertainties(process, metType="PF",
                                      correctionLevel=["T0","T1","T2","Smear","Txy"],
                                      computeUncertainties=False,
                                      produceIntermediateCorrections=True,
                                      addToPatDefaultSequence=False,
                                      jetCollectionUnskimmed=jetCollUnskimmed,
                                      photonCollection=photonColl,
                                      electronCollection=electronColl,
                                      muonCollection=muonColl,
                                      tauCollection=tauColl,
                                      pfCandCollection =pfCandColl,
                                      autoJetCleaning=jetCleaning,
                                      jecUncertaintyFile=jecUnFile,
                                      jetSelection=jetSelection,
                                      jetFlavor=jetFlavor,
                                      recoMetFromPFCs=recoMetFromPFCs,
                                      postfix=postfix
                                      )
    
    #MET T1 uncertainties
    runMETCorrectionsAndUncertainties(process, metType="PF",
                                      correctionLevel=["T1"],
                                      computeUncertainties=True,
                                      produceIntermediateCorrections=False,
                                      addToPatDefaultSequence=False,
                                      jetCollectionUnskimmed=jetCollUnskimmed,
                                      photonCollection=photonColl,
                                      electronCollection=electronColl,
                                      muonCollection=muonColl,
                                      tauCollection=tauColl,
                                      pfCandCollection =pfCandColl,
                                      autoJetCleaning=jetCleaning,
                                      jecUncertaintyFile=jecUnFile,
                                      jetSelection=jetSelection,
                                      jetFlavor=jetFlavor,
                                      recoMetFromPFCs=recoMetFromPFCs,
                                      postfix=postfix
                                      )
    
    #MET T1 Smeared JER uncertainties
    runMETCorrectionsAndUncertainties(process, metType="PF",
                                      correctionLevel=["T1","Smear"],
                                      computeUncertainties=True,
                                      produceIntermediateCorrections=False,
                                      addToPatDefaultSequence=False,
                                      jetCollectionUnskimmed=jetCollUnskimmed,
                                      photonCollection=photonColl,
                                      electronCollection=electronColl,
                                      muonCollection=muonColl,
                                      tauCollection=tauColl,
                                      pfCandCollection =pfCandColl,
                                      autoJetCleaning=jetCleaning,
                                      jecUncertaintyFile=jecUnFile,
                                      jetSelection=jetSelection,
                                      jetFlavor=jetFlavor,
                                      recoMetFromPFCs=recoMetFromPFCs,
                                      postfix=postfix,
                                      )




# miniAOD reproduction ===========================
def runMetCorAndUncFromMiniAOD(process, metType="PF",
                               jetCollUnskimmed="slimmedJets",
                               photonColl="slimmedPhotons",
                               electronColl="slimmedElectrons",
                               muonColl="slimmedMuons",
                               tauColl="slimmedTaus",
                               pfCandColl = "packedPFCandidates",
                               jetFlavor="AK4PFchs",
                               jetCleaning="LepClean",
                               isData=False,
                               manualJetConfig=False,
                               reclusterJets=False,
                               jetSelection="pt>15 && abs(eta)<9.9",
                               recoMetFromPFCs=False,
                               jetCorLabelL3="ak4PFCHSL1FastL2L3Corrector",
                               jetCorLabelRes="ak4PFCHSL1FastL2L3ResidualCorrector",
##                               jecUncFile="CondFormats/JetMETObjects/data/Summer15_50nsV5_DATA_UncertaintySources_AK4PFchs.txt",
                               CHS=False,
                               jecUncFile="",
                               postfix=""):

    runMETCorrectionsAndUncertainties = RunMETCorrectionsAndUncertainties()

    #MET T1 uncertainties
    runMETCorrectionsAndUncertainties(process, metType="PF",
                                      correctionLevel=["T1"],
                                      computeUncertainties=True,
                                      produceIntermediateCorrections=False,
                                      addToPatDefaultSequence=False,
                                      jetCollectionUnskimmed=jetCollUnskimmed,
                                      electronCollection=electronColl,
                                      muonCollection=muonColl,
                                      tauCollection=tauColl,
                                      photonCollection=photonColl,
                                      pfCandCollection =pfCandColl,
                                      runOnData=isData,
                                      onMiniAOD=True,
                                      reclusterJets=reclusterJets,
                                      jetSelection=jetSelection,
                                      recoMetFromPFCs=recoMetFromPFCs,
                                      autoJetCleaning=jetCleaning,
                                      manualJetConfig=manualJetConfig,
                                      jetFlavor=jetFlavor,
                                      jetCorLabelUpToL3=jetCorLabelL3,
                                      jetCorLabelL3Res=jetCorLabelRes,
                                      jecUncertaintyFile=jecUncFile,
                                      CHS=CHS,
                                      postfix=postfix,
                                      )
    
    #MET T1+Txy / Smear
    runMETCorrectionsAndUncertainties(process, metType="PF",
                                      correctionLevel=["T1","Txy"],
                                      computeUncertainties=False,
                                      produceIntermediateCorrections=True,
                                      addToPatDefaultSequence=False,
                                      jetCollectionUnskimmed=jetCollUnskimmed,
                                      electronCollection=electronColl,
                                      muonCollection=muonColl,
                                      tauCollection=tauColl,
                                      photonCollection=photonColl,
                                      pfCandCollection =pfCandColl,
                                      runOnData=isData,
                                      onMiniAOD=True,
                                      reclusterJets=reclusterJets,
                                      jetSelection=jetSelection,
                                      recoMetFromPFCs=recoMetFromPFCs,
                                      autoJetCleaning=jetCleaning,
                                      manualJetConfig=manualJetConfig,
                                      jetFlavor=jetFlavor,
                                      jetCorLabelUpToL3=jetCorLabelL3,
                                      jetCorLabelL3Res=jetCorLabelRes,
                                      jecUncertaintyFile=jecUncFile,
                                      CHS=CHS,
                                      postfix=postfix,
                                      )
    #MET T1+Smear + uncertainties
    runMETCorrectionsAndUncertainties(process, metType="PF",
                                      correctionLevel=["T1","Smear"],
                                      computeUncertainties=True,
                                      produceIntermediateCorrections=False,
                                      addToPatDefaultSequence=False,
                                      jetCollectionUnskimmed=jetCollUnskimmed,
                                      electronCollection=electronColl,
                                      muonCollection=muonColl,
                                      tauCollection=tauColl,
                                      photonCollection=photonColl,
                                      pfCandCollection =pfCandColl,
                                      runOnData=isData,
                                      onMiniAOD=True,
                                      reclusterJets=reclusterJets,
                                      jetSelection=jetSelection,
                                      recoMetFromPFCs=recoMetFromPFCs,
                                      autoJetCleaning=jetCleaning,
                                      manualJetConfig=manualJetConfig,
                                      jetFlavor=jetFlavor,
                                      jetCorLabelUpToL3=jetCorLabelL3,
                                      jetCorLabelL3Res=jetCorLabelRes,
                                      jecUncertaintyFile=jecUncFile,
                                      CHS=CHS,
                                      postfix=postfix,
                                      )
