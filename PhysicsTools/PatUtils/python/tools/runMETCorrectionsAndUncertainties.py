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
                          "Type of considered MET (only PF or MVA supported so far)", Type=str)
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
	self.addParameter(self._defaultParameters, 'jetCollection', cms.InputTag('selectedPatJets'),
                          "Input jet collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'jetCollectionUnskimmed', cms.InputTag('patJets'),
                          "Input unskimmed jet collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'pfCandCollection', cms.InputTag('particleFlow'),
                          "pf Candidate collection", Type=cms.InputTag, acceptNoneValue=True)

        self.addParameter(self._defaultParameters, 'jetCorrPayload', 'AK4PF',
                          "Use AK4PF/AK4PFCHS for PFJets,AK4Calo for CaloJets", Type=str)
        self.addParameter(self._defaultParameters, 'jetCorLabelUpToL3', cms.InputTag('ak4PFL1FastL2L3Corrector'), "Use ak4PFL1FastL2L3Corrector (ak4PFchsL1FastL2L3Corrector) for PFJets with (without) charged hadron subtraction, ak4CaloL1FastL2L3Corrector for CaloJets", Type=cms.InputTag)
        self.addParameter(self._defaultParameters, 'jetCorLabelL3Res', cms.InputTag('ak4PFL1FastL2L3ResidualCorrector'), "Use ak4PFL1FastL2L3ResidualCorrector (ak4PFchsL1FastL2L3ResiduaCorrectorl) for PFJets with (without) charged hadron subtraction, ak4CaloL1FastL2L3ResidualCorrector for CaloJets", Type=cms.InputTag)
        self.addParameter(self._defaultParameters, 'jecUncertaintyFile', 'PhysicsTools/PatUtils/data/Summer13_V1_DATA_UncertaintySources_AK5PF.txt',
                          "Extra JER uncertainty file", Type=str)
        self.addParameter(self._defaultParameters, 'jecUncertaintyTag', 'SubTotalMC',
                          "Extra JER uncertainty file", Type=str)
        
        self.addParameter(self._defaultParameters, 'addToPatDefaultSequence', True,
                          "Flag to enable/disable that metUncertaintySequence is inserted into patDefaultSequence", Type=bool)
        self.addParameter(self._defaultParameters, 'reclusterJets', False,
                  "Flag to enable/disable the jet reclustering", Type=bool)
        self.addParameter(self._defaultParameters, 'onMiniAOD', False,
                          "Switch on miniAOD configuration", Type=bool)
        self.addParameter(self._defaultParameters, 'repro74X', False,
                          "option for 74X miniAOD re-processing", Type=bool)          
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
                 jetCollection           =None,
                 jetCollectionUnskimmed  =None,
                 pfCandCollection        =None,
                 jetCorrPayload          =None,
                 jetCorLabelUpToL3       =None,
                 jetCorLabelL3Res        =None,
                 jecUncertaintyFile      =None,
                 jecUncertaintyTag       =None,
                 addToPatDefaultSequence =None,
                 reclusterJets           =None,
                 onMiniAOD               =None,
                 repro74X                =None,
                 postfix                 =None):
        electronCollection = self.initializeInputTag(electronCollection, 'electronCollection')
        photonCollection = self.initializeInputTag(photonCollection, 'photonCollection')
        muonCollection = self.initializeInputTag(muonCollection, 'muonCollection')
        tauCollection = self.initializeInputTag(tauCollection, 'tauCollection')
        jetCollection = self.initializeInputTag(jetCollection, 'jetCollection')
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
        if jetCollection is None :
            jetCollection = self._defaultParameters['jetCollection'].value
        if jetCollectionUnskimmed is None :
            jetCollectionUnskimmed = self._defaultParameters['jetCollectionUnskimmed'].value
        if pfCandCollection is None :
            pfCandCollection = self._defaultParameters['pfCandCollection'].value
        if jetCorrPayload is None :
            jetCorrPayload = self._defaultParameters['jetCorrPayload'].value
        if jetCorLabelUpToL3  is None:
            jetCorLabelUpToL3 = self._defaultParameters['jetCorLabelUpToL3'].value
        if jetCorLabelL3Res   is None:
            jetCorLabelL3Res = self._defaultParameters['jetCorLabelL3Res'].value
        if jecUncertaintyFile is None:
            jecUncertaintyFile = self._defaultParameters['jecUncertaintyFile'].value
        if jecUncertaintyTag  is None:
            jecUncertaintyTag = self._defaultParameters['jecUncertaintyTag'].value
            
        if addToPatDefaultSequence is None :
            addToPatDefaultSequence = self._defaultParameters['addToPatDefaultSequence'].value
        if reclusterJets is None :
            reclusterJets = self._defaultParameters['reclusterJets'].value
        if onMiniAOD is None :
            onMiniAOD = self._defaultParameters['onMiniAOD'].value
        if repro74X is None :
            repro74X = self._defaultParameters['repro74X'].value
        if postfix is None :
            postfix = self._defaultParameters['potsfix'].value

        self.setParameter('metType',metType),
        self.setParameter('correctionLevel',correctionLevel),
        self.setParameter('computeUncertainties',computeUncertainties),
        self.setParameter('produceIntermediateCorrections',produceIntermediateCorrections),
        self.setParameter('electronCollection',electronCollection),
        self.setParameter('photonCollection',photonCollection),
        self.setParameter('muonCollection',muonCollection),
        self.setParameter('tauCollection',tauCollection),
        self.setParameter('jetCollection',jetCollection),
        self.setParameter('jetCollectionUnskimmed',jetCollectionUnskimmed),
        self.setParameter('pfCandCollection',pfCandCollection),

        #jet energy scale uncertainty needs
        self.setParameter('jetCorrPayload',jetCorrPayload),
        self.setParameter('jetCorLabelUpToL3',jetCorLabelUpToL3),
        self.setParameter('jetCorLabelL3Res',jetCorLabelL3Res),
        #optional
        self.setParameter('jecUncertaintyFile',jecUncertaintyFile),
        self.setParameter('jecUncertaintyTag',jecUncertaintyTag),

        self.setParameter('addToPatDefaultSequence',addToPatDefaultSequence),
        self.setParameter('reclusterJets',reclusterJets),
        self.setParameter('onMiniAOD',onMiniAOD),
        self.setParameter('repro74X',repro74X),
        self.setParameter('postfix',postfix),
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
        jetCollection           = self._parameters['jetCollection'].value
        jetCollectionUnskimmed  = self._parameters['jetCollectionUnskimmed'].value
        pfCandCollection        = self._parameters['pfCandCollection'].value
        jetCorrPayload          = self._parameters['jetCorrPayload'].value
        jetCorLabelUpToL3       = self._parameters['jetCorLabelUpToL3'].value
        jetCorLabelL3Res        = self._parameters['jetCorLabelL3Res'].value
        jecUncertaintyFile      = self._parameters['jecUncertaintyFile'].value
        jecUncertaintyTag       = self._parameters['jecUncertaintyTag'].value

        addToPatDefaultSequence = self._parameters['addToPatDefaultSequence'].value
        reclusterJets           = self._parameters['reclusterJets'].value
        onMiniAOD               = self._parameters['onMiniAOD'].value
        repro74X                = self._parameters['repro74X'].value
        postfix                 = self._parameters['postfix'].value
        
        #prepare jet configuration
        jetUncInfos = { "jCorrPayload":jetCorrPayload, "jCorLabelUpToL3":jetCorLabelUpToL3,
                        "jCorLabelL3Res":jetCorLabelL3Res, "jecUncFile":jecUncertaintyFile,
                        "jecUncTag":jecUncertaintyTag }        

       
        #default MET production
        patMetModuleSequence = cms.Sequence()
        self.produceMET(process, metType,patMetModuleSequence, postfix)
              
        #preparation to run over miniAOD (met reproduction) 
        #-> could be extracted from the slimmedMET for a gain in CPU performances
        if onMiniAOD:
            reclusterJets = True
            self.miniAODConfiguration(process, 
                                      pfCandCollection,
                                      patMetModuleSequence,
                                      repro74X,
                                      postfix
                                      )

        #jet AK4 reclustering if needed for JECs
        if reclusterJets:
            jetCollection = self.ak4JetReclustering(process, pfCandCollection, 
                                                    patMetModuleSequence, postfix)

        # correct the MET
        patMetCorrectionSequence, metModName = self.getCorrectedMET(process, metType, correctionLevel,
                                                                    produceIntermediateCorrections,
                                                                    patMetModuleSequence, postfix )

        #fix the default jets for the type1 computation to those used to compute the uncertainties
        #in order to be consistent with what is done in the correction and uncertainty step
        #particularly true for miniAODs
        if isValidInputTag(jetCollectionUnskimmed) and "T1" in metModName:
            getattr(process,"patPFMetT1T2Corr").src = jetCollectionUnskimmed
            getattr(process,"patPFMetT2Corr").src = jetCollectionUnskimmed
       
        #compute the uncertainty on the MET
        patMetUncertaintySequence = cms.Sequence()
        if computeUncertainties:
            patMetUncertaintySequence =  self.getMETUncertainties(process, metType, metModName,
                                                                  electronCollection,
                                                                  photonCollection,
                                                                  muonCollection,
                                                                  tauCollection,
                                                                  jetCollection,
                                                                  jetUncInfos,
                                                                  patMetModuleSequence,
                                                                  postfix)

      
        setattr(process, "patMetCorrectionSequence"+postfix, patMetCorrectionSequence)
        setattr(process, "patMetUncertaintySequence"+postfix, patMetUncertaintySequence)
        setattr(process, "patMetModuleSequence"+postfix, patMetModuleSequence)
        
        #prepare and fill the final sequence containing all the sub-sequence
        fullPatMetSequence = cms.Sequence()
        fullPatMetSequence += getattr(process, "patMetCorrectionSequence"+postfix)
        fullPatMetSequence += getattr(process, "patMetUncertaintySequence"+postfix)
        fullPatMetSequence += getattr(process, "patMetModuleSequence"+postfix)
                
        setattr(process,"fullPatMetSequence"+postfix,fullPatMetSequence)

        # insert the fullPatMetSequence into patDefaultSequence if needed
        if addToPatDefaultSequence:
            if not hasattr(process, "patDefaultSequence"):
                raise ValueError("PAT default sequence is not defined !!")
            process.patDefaultSequence += getattr(process, "fullPatMetSequence"+postfix)
    
#====================================================================================================
    def produceMET(self, process,  metType, metModuleSequence, postfix):
        if not hasattr(process, 'pat'+metType+'Met'):
            process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")
            if postfix != "":
                setattr(process, 'pat'+metType+'Met'+postfix, getattr(process,'pat'+metType+'Met' ).clone() )
            metModuleSequence += getattr(process, 'pat'+metType+'Met'+postfix )

#====================================================================================================
    def getCorrectedMET(self, process, metType, correctionLevel,produceIntermediateCorrections, metModuleSequence, postfix ):
        
        # default outputs
        patMetCorrectionSequence = cms.Sequence()
        metModName = "pat"+metType+"Met"+postfix
       
        # loading correction file if not already here
        if not hasattr(process, 'patMetCorrectionSequence'):
            process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")

     ## MM: FIXME, smearing procedure needs a lot of work, still 2010 recipes everywhere
     ## ==> smearing disabled for all cases
     #   if cor == "Smear":
     #       #print "WARNING: smearing procedure still uses 2010 recipe, disabled per default for the moment"
            

        corNames = { #not really needed but in case we have changes in the future....
            "T0":"T0pc",
            "T1":"T1",
            "T2":"T2",
            "Txy":"Txy",
            "Smear":"Smear",
            }
        
        
        #if empty correction level, no need to try something
        for cor in correctionLevel: #MM to be changed!!!!!!
            if cor not in corNames.keys():
                #print "ERROR : ",cor," is not a proper MET correction name! aborting the MET correction production"
                return patMetCorrectionSequence, metModName

        corModNames = {
            "T0": "patPFMetT0CorrSequence"+postfix,
            "T1": "patPFMetT1T2CorrSequence"+postfix,
            "T2": "patPFMetT2CorrSequence"+postfix,
            "Txy": "patPFMetTxyCorrSequence"+postfix,
            "Smear": "patPFMetSmearCorrSequence"+postfix,
            "T2Smear": "patPFMetT2SmearCorrSequence"+postfix
            }

        corModules = {}
        for mod in corModNames.keys():
            corModules[mod] = getattr(process, corModNames[mod] )
                  
        corTags = {
            "T0":cms.InputTag('patPFMetT0Corr'+postfix),
            "T1":cms.InputTag('patPFMetT1T2Corr'+postfix, 'type1'),
            "T2":cms.InputTag('patPFMetT2Corr'+postfix,   'type2'),
            "Txy": cms.InputTag('patPFMetTxyCorr'+postfix),
            "Smear":cms.InputTag('patPFMetSmearCorr'+postfix, 'type1'),
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
          #  if not produceIntermediateCorrections:
          #      #print "REMOVAL"
         #   correctionSequence.remove( corModules["Smear"] )
         #   corrections.remove(corTags["Smear"])


        #Txy parameter tuning
        if "Txy" in correctionLevel:
            self.tuneTxyParameters(process, corScheme, postfix)
     
        #Enable MET significance in the type1 MET is computed
        #if "T1" in correctionLevel:
        #    getattr(process, "pat"+metType+"Met"+postfix).computeMETSignificance = cms.bool(True)


        #create the main MET producer
        metModName = "pat"+metType+"Met"+corScheme+postfix
        corMetProducer = cms.EDProducer("CorrectedPATMETProducer",
                 src = cms.InputTag('pat'+metType+'Met' + postfix),
                 srcCorrections = cms.VInputTag(corrections)
               )
        setattr(process,metModName, corMetProducer)

        # adding the full sequence only if it does not exist
        if not hasattr(process, 'patMetCorrectionSequence'+postfix):
            for corModule in correctionSequence:
                patMetCorrectionSequence += corModule
            setattr(process, "patMetCorrectionSequence"+postfix, patMetCorrectionSequence)
            
        else: #if it exists, only add the missing correction modules, no need to redo everything
            patMetCorrectionSequence = cms.Sequence()
            setattr(process, "patMetCorrectionSequence"+postfix,patMetCorrectionSequence)
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
    def getMETUncertainties(self, process, metType, metModName, electronCollection, photonCollection,
                            muonCollection, tauCollection, jetCollection, jetUncInfos, patMetModuleSequence, postfix):

        
        # uncertainty sequence
        metUncSequence = cms.Sequence()

        #===================================================================================
        # jet energy resolution shifts
        #===================================================================================
        if not isValidInputTag(jetCollection): #or jetCollection=="":
            print "INFO : jet collection %s does not exists, no energy resolution shifting will be performed in MET uncertainty tools" % jetCollection
        else: 
            preId=""
            if "Smear" in metModName:
                preId="Smeared"

            metJERUncModules = self.getVariations(process, metModName, "Jet",preId, jetCollection, "Res", metUncSequence, postfix )
            
            for mod in metJERUncModules.keys():
                setattr(process, mod, metJERUncModules[mod] )
                patMetModuleSequence += getattr(process, mod)

        #===================================================================================
        # Unclustered energy shifts
        #===================================================================================
        metUnclEUncModules = self.getUnclusteredVariations(process, metModName, metUncSequence, postfix )
        for mod in metUnclEUncModules.keys():
            setattr(process, mod, metUnclEUncModules[mod] )
            patMetModuleSequence += getattr(process, mod)

        #===================================================================================
        # Other energy shifts
        #===================================================================================
        objectCollections = { "Jet":jetCollection,
                              "Electron":electronCollection,
                              "Photon":photonCollection,
                              "Muon":muonCollection,
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
                    patMetModuleSequence += getattr(process, mod)

        #return the sequence containing the shifted collections producers
        return metUncSequence

#====================================================================================================
    def createEnergyScaleShiftedUpModule(self, process,identifier, objectCollection,
                                         varyByNsigmas, jetUncInfos=None, postfix=""):

        shiftedModuleUp = None
        
        if identifier == "Electron":
            shiftedModuleUp = cms.EDProducer("ShiftedPATElectronProducer",
                                             src = objectCollection,
                                             binning = cms.VPSet(
                    cms.PSet(
                        binSelection = cms.string('isEB'),
                        binUncertainty = cms.double(0.006)
                        ),
                    cms.PSet(
                        binSelection = cms.string('!isEB'),
                        binUncertainty = cms.double(0.015)
                        ),
                    ),
                                             shiftBy = cms.double(+1.*varyByNsigmas)
                                             )
            
        if identifier == "Photon":
            shiftedModuleUp = cms.EDProducer("ShiftedPATPhotonProducer",
                                             src = objectCollection,
                                             binning = cms.VPSet(
                    cms.PSet(
                        binSelection = cms.string('isEB'),
                        binUncertainty = cms.double(0.01)
                        ),
                    cms.PSet(
                        binSelection = cms.string('!isEB'),
                        binUncertainty = cms.double(0.025)
                        ),
                    ),
                                             shiftBy = cms.double(+1.*varyByNsigmas)
                                             )

        if identifier == "Muon":
            shiftedModuleUp = cms.EDProducer("ShiftedPATMuonProducer",
                                             src = objectCollection,
                                             binning = cms.VPSet(
                    cms.PSet(
                        binSelection = cms.string('pt < 100'),
                        binUncertainty = cms.double(0.002)
                        ),
                    cms.PSet(
                        binSelection = cms.string('pt >= 100'),
                        binUncertainty = cms.double(0.05)
                        ),
                    ),
                                             shiftBy = cms.double(+1.*varyByNsigmas)
                                             )
            
        if identifier == "Tau":
            shiftedModuleUp = cms.EDProducer("ShiftedPATTauProducer",
                                             src = objectCollection,
                                             uncertainty = cms.double(0.03),
                                             shiftBy = cms.double(+1.*varyByNsigmas)
                                             )

        if identifier == "Jet":
            shiftedModuleUp = cms.EDProducer("ShiftedPATJetProducer",
                                             src = objectCollection,
                                             jetCorrInputFileName = cms.FileInPath(jetUncInfos["jecUncFile"] ), #jecUncertaintyFile),
                                             jetCorrUncertaintyTag = cms.string(jetUncInfos["jecUncTag"] ), #jecUncertaintyTag),
                                             addResidualJES = cms.bool(True),
                                             jetCorrLabelUpToL3 = cms.InputTag(jetUncInfos["jCorLabelUpToL3"].value() ), #jetCorrLabelUpToL3.value()),
                                             jetCorrLabelUpToL3Res = cms.InputTag(jetUncInfos["jCorLabelL3Res"].value() ), #jetCorrLabelUpToL3Res.value()),
                                             shiftBy = cms.double(+1.*varyByNsigmas)
                                             )

        return shiftedModuleUp


#====================================================================================================
    def removePostfix(self, name, postfix):
        
        if postfix=="":
            return name

        baseName = name
        if baseName[-len(postfix):] == postfix:
            baseName = baseName[0:-len(postfix)]
        else:
            raise StandardError("Tried to remove postfix %s from %s, but it wasn't there" % (postfix, baseName))
        
        return name

#====================================================================================================
    def tuneTxyParameters(self, process, corScheme, postfix):
        import PhysicsTools.PatUtils.patPFMETCorrections_cff as metCors
        xyTags = {
            "Txy":metCors.patMultPhiCorrParams_Txy,
            "T1Txy":metCors.patMultPhiCorrParams_T1Txy,
            "T0pcTxy":metCors.patMultPhiCorrParams_T0pcTxy,
            "T0pcT1Txy":metCors.patMultPhiCorrParams_T0pcT1Txy,
            "T1T2Txy":metCors.patMultPhiCorrParams_T1T2Txy,
            "T0pcT1T2Txy":metCors.patMultPhiCorrParams_T0pcT1T2Txy,
            "T1SmearTxy":metCors.patMultPhiCorrParams_T1SmearTxy,
            "T1T2SmearTxy":metCors.patMultPhiCorrParams_T1T2SmearTxy,
            "T0pcT1SmearTxy":metCors.patMultPhiCorrParams_T0pcT1SmearTxy,
            "T0pcT1T2SmearTxy":metCors.patMultPhiCorrParams_T0pcT1T2SmearTxy
            }
        
        getattr(process, "patPFMetTxyCorr"+postfix).parameters = xyTags[corScheme] 




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
                objectCollection=cms.InputTag("selectedPatJetsForMetT1T2SmearCorr")

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
            setattr(process, modName, shiftedCollModules[mod])
            metUncSequence += getattr(process, modName)
            
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

        return shiftedMetProducers


#========================================================================================
    def createShiftedMETModule(self, process, originCollection, shiftedCollection):

        shiftedModule = cms.EDProducer("ShiftedParticleMETcorrInputProducer",
                                       srcOriginal = originCollection,
                                       srcShifted = cms.InputTag(shiftedCollection),
                                       )

        return shiftedModule

#========================================================================================
    def getUnclusteredVariations(self, process, metModName, metUncSequence, postfix ):

        varyByNsigmas=1

        unclEnMETcorrectionsSrcs = [
            [ 'pfCandMETcorr' + postfix, [ '' ] ],
            [ 'patPFMetT1T2Corr' + postfix, [ 'type2', 'offset' ] ],
            [ 'patPFMetT2Corr' + postfix, [ 'type2' ] ],
            ]
        
        #MM missing protection against missing corrections needed to compute the uncertainties 
        #for srcUnclEnMETcorr in unclEnMETcorrectionsSrcs:
        #    if not hasattr(process, srcUnclEnMETcorr[0])
        #    metUncSequence

        shiftedMetProducers = {}

        variations={"Up":1.,"Down":-1.}
        for var in variations.keys():
            
            for srcUnclEnMETcorr in unclEnMETcorrectionsSrcs:
                moduleUnclEnMETcorr = cms.EDProducer("ShiftedMETcorrInputProducer",
                                                       src = cms.VInputTag(
                        [ cms.InputTag(srcUnclEnMETcorr[0], instanceLabel) for instanceLabel in srcUnclEnMETcorr[1] ]
                        ),
                                                     uncertainty = cms.double(0.10),
                                                     shiftBy = cms.double(variations[var]*varyByNsigmas)
                                                     )
                
                baseName = self.removePostfix(srcUnclEnMETcorr[0], postfix)
              
                moduleUnclEnMETcorrName = baseName+"UnclusteredEn"+var+postfix
                setattr(process, moduleUnclEnMETcorrName, moduleUnclEnMETcorr)
                metUncSequence += moduleUnclEnMETcorr
                unclEnMETcorrections = ([ cms.InputTag(moduleUnclEnMETcorrName, instanceLabel)
                                          for instanceLabel in srcUnclEnMETcorr[1] ] )

            #and finally prepare the shifted MET producers
            modName = self.removePostfix(metModName, postfix)
            modName = modName+"UnclusteredEn"+var+postfix
            shiftedMETModule = getattr(process, metModName).clone(
                src = cms.InputTag( metModName ),
                srcCorrections = cms.VInputTag( unclEnMETcorrections )
                )
            shiftedMetProducers[ modName ] = shiftedMETModule

        return shiftedMetProducers

#========================================================================================
    def createSmearedJetModule(self, process, jetCollection, smear, varyByNsigmas, varDir, metUncSequence, postfix):
        
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


        setattr(process, modName, getattr(process, "patSmearedJets").clone(
                src = jetCollection,
                areSrcJetsSmeared = cms.bool(smear),
                shiftBy = cms.double(varyByNsigmas),
                ) )    
        metUncSequence += getattr(process, modName)
      
        smearedJetModule = getattr(process, "selectedPatJetsForMetT1T2SmearCorr").clone(
                src = cms.InputTag(modName)
        )
        
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



    def ak4JetReclustering(self,process, pfCandCollection, patMetModuleSequence, postfix):
        
        if not hasattr(process, "ak4PFJets"):
            process.load("RecoJets.JetProducers.ak4PFJets_cfi")
            print "reclustering ak4pf", pfCandCollection
            process.ak4PFJets.src = pfCandCollection 
            process.ak4PFJets.doAreaFastjet = True
            
            patMetModuleSequence += getattr(process, "ak4PFJets")
            
            switchJetCollection(process,
                                jetSource = cms.InputTag('ak4PFJets'),
                                jetCorrections = ('AK4PF', ['L1FastJet', 'L2Relative', 'L3Absolute'], '')
                                )

            process.patJets.addGenJetMatch = False 
            process.patJets.addGenPartonMatch = False 
            process.patJets.addPartonJetMatch = False 
            del process.patJets.JetFlavourInfoSource
            del process.patJets.JetPartonMapSource
            process.patJets.getJetMCFlavour = False
            
            process.patJetCorrFactors.primaryVertices= cms.InputTag("offlineSlimmedPrimaryVertices")

        return cms.InputTag("selectedPatJets")
        

    def miniAODConfiguration(self, process, pfCandCollection, patMetModuleSequence, repro74X, postfix ):
      
        if not hasattr(process, "pfMet"):
            process.load("RecoMET.METProducers.PFMET_cfi")
            process.pfMet.src = pfCandCollection
            process.pfMet.calculateSignificance = False

            patMetModuleSequence += getattr(process, "pfMet")

            getattr(process, "patPFMet").addGenMET  = False
     
            getattr(process, "patPFMetTxyCorr").srcPFlow = pfCandCollection
            getattr(process, "patPFMetTxyCorr").vertexCollection = cms.InputTag("offlineSlimmedPrimaryVertices")

        if not hasattr(process, "slimmedMETs"):
            process.load("PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi")
            process.slimmedMETs.src = cms.InputTag("patPFMetT1")
            process.slimmedMETs.runningOnMiniAOD = True
            process.slimmedMETs.t01Variation = cms.InputTag("slimmedMETs","","PAT")
         
            #smearing and type0 variations not yet supported in reprocessing
            del process.slimmedMETs.t1SmearedVarsAndUncs
            del process.slimmedMETs.tXYUncForT01
            del process.slimmedMETs.tXYUncForT1Smear
            del process.slimmedMETs.tXYUncForT01Smear
            del process.slimmedMETs.caloMET

            if repro74X:
                del process.slimmedMETs.t01Variation

#========================================================================================
runMETCorrectionsAndUncertainties = RunMETCorrectionsAndUncertainties()
