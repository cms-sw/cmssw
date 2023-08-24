from __future__ import print_function
import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.ConfigToolBase import *
import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask, addTaskToProcess
from PhysicsTools.PatAlgos.tools.jetTools import switchJetCollection
import CommonTools.CandAlgos.candPtrProjector_cfi as _mod
from PhysicsTools.PatUtils.tools.pfforTrkMET_cff import *
import JetMETCorrections.Type1MET.BadPFCandidateJetsEEnoiseProducer_cfi as _modbad
import JetMETCorrections.Type1MET.UnclusteredBlobProducer_cfi as _modunc

# function to determine whether a valid input tag was given
def isValidInputTag(input):
    input_str = input
    if isinstance(input, cms.InputTag):
        input_str = input.value()
    if input is None or input_str == '""':
        return False
    else:
        return True

# class to manage the (re-)calculation of MET, its corrections, and its uncertainties
class RunMETCorrectionsAndUncertainties(ConfigToolBase):

    _label='RunMETCorrectionsAndUncertainties'
    _defaultParameters=dicttypes.SortedKeysDict()

    def __init__(self):
        ConfigToolBase.__init__(self)
        # MET type, correctionlevel, and uncertainties
        self.addParameter(self._defaultParameters, 'metType', "PF",
                          "Type of considered MET (only PF and Puppi supported so far)", Type=str, allowedValues = ["PF","Puppi"])
        self.addParameter(self._defaultParameters, 'correctionLevel', [""],
                          "level of correction : available corrections for pfMet are T0, T1, T2, Txy and Smear)",
                          allowedValues=["T0","T1","T2","Txy","Smear",""])
        self.addParameter(self._defaultParameters, 'computeUncertainties', True,
                          "enable/disable the uncertainty computation", Type=bool)
        self.addParameter(self._defaultParameters, 'produceIntermediateCorrections', False,
                          "enable/disable the production of all correction schemes (only for the most common)", Type=bool)

        # high-level object collections used e.g. for MET uncertainties or jet cleaning
        self.addParameter(self._defaultParameters, 'electronCollection', cms.InputTag('selectedPatElectrons'),
                          "Input electron collection", Type=cms.InputTag, acceptNoneValue=True)
        self.addParameter(self._defaultParameters, 'photonCollection', cms.InputTag('selectedPatPhotons'),
                          "Input photon collection", Type=cms.InputTag, acceptNoneValue=True)
        self.addParameter(self._defaultParameters, 'muonCollection', cms.InputTag('selectedPatMuons'),
                          "Input muon collection", Type=cms.InputTag, acceptNoneValue=True)
        self.addParameter(self._defaultParameters, 'tauCollection', cms.InputTag('selectedPatTaus'),
                          "Input tau collection", Type=cms.InputTag, acceptNoneValue=True)
        self.addParameter(self._defaultParameters, 'jetCollectionUnskimmed', cms.InputTag('patJets'),
                          "Input unskimmed jet collection for T1 MET computation", Type=cms.InputTag, acceptNoneValue=True)

        # pf candidate collection used for recalculation of MET from pf candidates
        self.addParameter(self._defaultParameters, 'pfCandCollection', cms.InputTag('particleFlow'),
                          "pf Candidate collection", Type=cms.InputTag, acceptNoneValue=True)

        # some options influencing MET corrections and uncertainties calculation
        self.addParameter(self._defaultParameters, 'autoJetCleaning', 'LepClean',
                          "Enable the jet cleaning for the uncertainty computation: Full for tau/photons/jet cleaning, Partial for jet cleaning, LepClean for jet cleaning with muon and electrons only, None or Manual for no cleaning",
                          allowedValues = ["Full","Partial","LepClean","None"])
        self.addParameter(self._defaultParameters, 'jetFlavor', 'AK4PFchs',
                          "Use AK4PF/AK4PFchs for PFJets,AK4Calo for CaloJets", Type=str, allowedValues = ["AK4PF","AK4PFchs","AK4PFPuppi","CaloJets"])
        self.addParameter(self._defaultParameters, 'jetCorrectionType', 'L1L2L3-L1',
                          "Use L1L2L3-L1 for the standard L1 removal / L1L2L3-RC for the random-cone correction", Type=str, allowedValues = ["L1L2L3-L1","L1L2L3-RC"])

        # technical options determining which JES corrections are used
        self.addParameter(self._defaultParameters, 'jetCorLabelUpToL3', "ak4PFCHSL1FastL2L3Corrector", "Use ak4PFL1FastL2L3Corrector (ak4PFCHSL1FastL2L3Corrector) for PFJets with (without) charged hadron subtraction, ak4CaloL1FastL2L3Corrector for CaloJets", Type=str)
        self.addParameter(self._defaultParameters, 'jetCorLabelL3Res', "ak4PFCHSL1FastL2L3ResidualCorrector", "Use ak4PFL1FastL2L3ResidualCorrector (ak4PFCHSL1FastL2L3ResidualCorrector) for PFJets with (without) charged hadron subtraction, ak4CaloL1FastL2L3ResidualCorrector for CaloJets", Type=str)
        # the file is used only for local running
        self.addParameter(self._defaultParameters, 'jecUncertaintyFile', '',
                          "Extra JES uncertainty file", Type=str)
        self.addParameter(self._defaultParameters, 'jecUncertaintyTag', None,
                          "JES uncertainty Tag", acceptNoneValue=True) # Type=str,

        # options to apply selections to the considered jets
        self.addParameter(self._defaultParameters, 'manualJetConfig', False,
                  "Enable jet configuration options", Type=bool)
        self.addParameter(self._defaultParameters, 'jetSelection', 'pt>15 && abs(eta)<9.9',
                          "Advanced jet kinematic selection", Type=str)

        # flags to influence how the MET is (re-)calculated, e.g. completely from scratch or just propagating new JECs
        self.addParameter(self._defaultParameters, 'recoMetFromPFCs', False,
                  "Recompute the MET from scratch using the pfCandidate collection", Type=bool)
        self.addParameter(self._defaultParameters, 'reapplyJEC', True,
                  "Flag to enable/disable JEC update", Type=bool)
        self.addParameter(self._defaultParameters, 'reclusterJets', False,
                  "Flag to enable/disable the jet reclustering", Type=bool)
        self.addParameter(self._defaultParameters, 'computeMETSignificance', True,
                  "Flag to enable/disable the MET significance computation", Type=bool)
        self.addParameter(self._defaultParameters, 'CHS', False,
                  "Flag to enable/disable the CHS jets", Type=bool)

        # information on what dataformat or datatype we are running on
        self.addParameter(self._defaultParameters, 'runOnData', False,
                          "Switch for data/MC processing", Type=bool)
        self.addParameter(self._defaultParameters, 'onMiniAOD', False,
                          "Switch on miniAOD configuration", Type=bool)

        # special input parameters when running over 2017 data
        self.addParameter(self._defaultParameters,'fixEE2017', False,
                          "Exclude jets and PF candidates with EE noise characteristics (fix for 2017 run)", Type=bool)
        self.addParameter(self._defaultParameters,'fixEE2017Params', {'userawPt': True, 'ptThreshold': 50.0, 'minEtaThreshold': 2.65, 'maxEtaThreshold': 3.139},
                          "Parameters dict for fixEE2017: userawPt, ptThreshold, minEtaThreshold, maxEtaThreshold", Type=dict)
        self.addParameter(self._defaultParameters, 'extractDeepMETs', False,
                          "Extract DeepMETs from miniAOD, instead of recomputing them.", Type=bool)

        # technical parameters
        self.addParameter(self._defaultParameters, 'Puppi', False,
                          "Puppi algorithm (private)", Type=bool)
        self.addParameter(self._defaultParameters, 'puppiProducerLabel', 'puppi',
                          "PuppiProducer module for jet clustering label name", Type=str)
        self.addParameter(self._defaultParameters, 'puppiProducerForMETLabel', 'puppiNoLep',
                          "PuppiProducer module for MET clustering label name", Type=str)
        self.addParameter(self._defaultParameters, 'addToPatDefaultSequence', False,
                          "Flag to enable/disable that metUncertaintySequence is inserted into patDefaultSequence", Type=bool)
        self.addParameter(self._defaultParameters, 'postfix', '',
                          "Technical parameter to identify the resulting sequences/tasks and its corresponding modules (allows multiple calls in a job)", Type=str)

        self.addParameter(self._defaultParameters, 'campaign', '', 'Production campaign', Type=str)
        self.addParameter(self._defaultParameters, 'era', '', 'Era e.g. 2018, 2017B, ...', Type=str)
        # make another parameter collection by copying the default parameters collection
        # later adapt the newly created parameter collection to always have a copy of the default parameters
        self._parameters = copy.deepcopy(self._defaultParameters)
        self._comment = ""

    # function to return the saved default parameters of the class
    def getDefaultParameters(self):
        return self._defaultParameters

#=========================================================================================
# implement __call__ function to be able to use class instances like functions
# reads the given parameters and saves them
# runs the toolCode function in the end
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
                 addToPatDefaultSequence =None,
                 manualJetConfig         =None,
                 jetSelection            =None,
                 recoMetFromPFCs         =None,
                 reapplyJEC              =None,
                 reclusterJets           =None,
                 computeMETSignificance  =None,
                 CHS                     =None,
                 puppiProducerLabel      =None,
                 puppiProducerForMETLabel = None,
                 runOnData               =None,
                 onMiniAOD               =None,
                 fixEE2017               =None,
                 fixEE2017Params         =None,
                 extractDeepMETs         =None,
                 campaign                =None,
                 era                     =None,
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
        if addToPatDefaultSequence is None :
            addToPatDefaultSequence = self._defaultParameters['addToPatDefaultSequence'].value
        if manualJetConfig is None :
            manualJetConfig =  self._defaultParameters['manualJetConfig'].value
        if jetSelection is None :
            jetSelection = self._defaultParameters['jetSelection'].value
        recoMetFromPFCsIsNone = (recoMetFromPFCs is None)
        if recoMetFromPFCs is None :
            recoMetFromPFCs =  self._defaultParameters['recoMetFromPFCs'].value
        if reapplyJEC is None :
            reapplyJEC = self._defaultParameters['reapplyJEC'].value
        reclusterJetsIsNone = (reclusterJets is None)
        if reclusterJets is None :
            reclusterJets = self._defaultParameters['reclusterJets'].value
        if computeMETSignificance is None :
            computeMETSignificance = self._defaultParameters['computeMETSignificance'].value
        if CHS is None :
            CHS = self._defaultParameters['CHS'].value
        if puppiProducerLabel is None:
            puppiProducerLabel = self._defaultParameters['puppiProducerLabel'].value
        if puppiProducerForMETLabel is None:
            puppiProducerForMETLabel = self._defaultParameters['puppiProducerForMETLabel'].value
        if runOnData is None :
            runOnData = self._defaultParameters['runOnData'].value
        if onMiniAOD is None :
            onMiniAOD = self._defaultParameters['onMiniAOD'].value
        if postfix is None :
            postfix = self._defaultParameters['postfix'].value
        if fixEE2017 is None :
            fixEE2017 = self._defaultParameters['fixEE2017'].value
        if fixEE2017Params is None :
            fixEE2017Params = self._defaultParameters['fixEE2017Params'].value
        if extractDeepMETs is None :
            extractDeepMETs = self._defaultParameters['extractDeepMETs'].value
        if campaign is None :
            campaign = self._defaultParameters['campaign'].value
        if era is None :
            era = self._defaultParameters['era'].value

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

        self.setParameter('addToPatDefaultSequence',addToPatDefaultSequence),
        self.setParameter('jetSelection',jetSelection),
        self.setParameter('recoMetFromPFCs',recoMetFromPFCs),
        self.setParameter('reclusterJets',reclusterJets),
        self.setParameter('computeMETSignificance',computeMETSignificance),
        self.setParameter('reapplyJEC',reapplyJEC),
        self.setParameter('CHS',CHS),
        self.setParameter('puppiProducerLabel',puppiProducerLabel),
        self.setParameter('puppiProducerForMETLabel',puppiProducerForMETLabel),
        self.setParameter('runOnData',runOnData),
        self.setParameter('onMiniAOD',onMiniAOD),
        self.setParameter('postfix',postfix),
        self.setParameter('fixEE2017',fixEE2017),
        self.setParameter('fixEE2017Params',fixEE2017Params),
        self.setParameter('extractDeepMETs',extractDeepMETs),
        self.setParameter('campaign',campaign),
        self.setParameter('era',era),

        # if puppi MET, autoswitch to std jets
        if metType == "Puppi":
            self.setParameter('CHS',False),

        # enabling puppi flag
        # metType is set back to PF because the necessary adaptions are handled via a postfix and directly changing parameters when Puppi parameter is true
        self.setParameter('Puppi',self._defaultParameters['Puppi'].value)
        if metType == "Puppi":
            self.setParameter('metType',"PF")
            self.setParameter('Puppi',True)

        # jet energy scale uncertainty needs
        if manualJetConfig:
            self.setParameter('CHS',CHS)
            self.setParameter('jetCorLabelUpToL3',jetCorLabelUpToL3)
            self.setParameter('jetCorLabelL3Res',jetCorLabelL3Res)
            self.setParameter('reclusterJets',reclusterJets)
        else:
            # internal jet configuration
            self.jetConfiguration()

        # defaults for 2017 fix
        # (don't need to recluster, just uses a subset of the input jet coll)
        if fixEE2017:
            if recoMetFromPFCsIsNone: self.setParameter('recoMetFromPFCs',True)
            if reclusterJetsIsNone: self.setParameter('reclusterJets',False)

        # met reprocessing and jet reclustering
        if recoMetFromPFCs and reclusterJetsIsNone and not fixEE2017:
            self.setParameter('reclusterJets',True)

        self.apply(process)


    def toolCode(self, process):
        ################################
        ### 1. read given parameters ###
        ################################

        # MET type, corrections, and uncertainties
        metType                 = self._parameters['metType'].value
        correctionLevel         = self._parameters['correctionLevel'].value
        computeUncertainties    = self._parameters['computeUncertainties'].value
        produceIntermediateCorrections = self._parameters['produceIntermediateCorrections'].value
        # physics object collections to consider when recalculating MET
        electronCollection      = self._parameters['electronCollection'].value
        photonCollection        = self._parameters['photonCollection'].value
        muonCollection          = self._parameters['muonCollection'].value
        tauCollection           = self._parameters['tauCollection'].value
        jetCollectionUnskimmed  = self._parameters['jetCollectionUnskimmed'].value
        pfCandCollection        = self._parameters['pfCandCollection'].value
        # jet specific options: jet corrections/uncertainties as well as jet selection/cleaning options
        jetSelection            = self._parameters['jetSelection'].value
        autoJetCleaning         = self._parameters['autoJetCleaning'].value
        jetFlavor               = self._parameters['jetFlavor'].value
        jetCorLabelUpToL3       = self._parameters['jetCorLabelUpToL3'].value
        jetCorLabelL3Res        = self._parameters['jetCorLabelL3Res'].value
        jecUncertaintyFile      = self._parameters['jecUncertaintyFile'].value
        jecUncertaintyTag       = self._parameters['jecUncertaintyTag'].value
        # additional MET calculation/extraction options
        recoMetFromPFCs         = self._parameters['recoMetFromPFCs'].value
        reapplyJEC              = self._parameters['reapplyJEC'].value
        reclusterJets           = self._parameters['reclusterJets'].value
        computeMETSignificance  = self._parameters['computeMETSignificance'].value
        extractDeepMETs         = self._parameters['extractDeepMETs'].value
        # specific option for 2017 EE noise mitigation
        fixEE2017               = self._parameters['fixEE2017'].value
        fixEE2017Params         = self._parameters['fixEE2017Params'].value
        campaign                = self._parameters['campaign'].value
        era                     = self._parameters['era'].value
        # additional runtime options
        onMiniAOD               = self._parameters['onMiniAOD'].value
        addToPatDefaultSequence = self._parameters['addToPatDefaultSequence'].value
        postfix                 = self._parameters['postfix'].value

        # prepare jet configuration used during MET (re-)calculation
        jetUncInfos = {
                        "jCorrPayload":jetFlavor,
                        "jCorLabelUpToL3":jetCorLabelUpToL3,
                        "jCorLabelL3Res":jetCorLabelL3Res,
                        "jecUncFile":jecUncertaintyFile,
                        "jecUncTag":"Uncertainty"
        }

        # get jet uncertainties from file
        if (jecUncertaintyFile!="" and jecUncertaintyTag==None):
            jetUncInfos[ "jecUncTag" ] = ""
        # get jet uncertainties from tag
        elif (jecUncertaintyTag!=None):
            jetUncInfos[ "jecUncTag" ] = jecUncertaintyTag

        #############################
        ### 2. (re-)construct MET ###
        #############################

        # task for main MET construction modules
        patMetModuleTask = cms.Task()

        # 2017 EE fix will modify pf cand and jet collections used downstream
        if fixEE2017:
            pfCandCollection, jetCollectionUnskimmed = self.runFixEE2017(process,
                fixEE2017Params,
                jetCollectionUnskimmed,
                pfCandCollection,
                [electronCollection,muonCollection,tauCollection,photonCollection],
                patMetModuleTask,
                postfix,
            )

        # recompute the MET (and thus the jets as well for correction) from scratch i.e. from pfcandidates
        if recoMetFromPFCs:
            self.recomputeRawMetFromPfcs(process,
                                         pfCandCollection,
                                         onMiniAOD,
                                         patMetModuleTask,
                                         postfix)
        # if not using pfcandidates, you also can extract the raw MET from MiniAOD
        elif onMiniAOD:
            self.extractMET(process, "raw", patMetModuleTask, postfix)

        # jet AK4 reclustering if needed for JECs ...
        if reclusterJets:
            jetCollectionUnskimmed = self.ak4JetReclustering(process, pfCandCollection,
                                                             patMetModuleTask, postfix)

        # ... or reapplication of JECs to already existing jets in MiniAOD
        if onMiniAOD:
            if not reclusterJets and reapplyJEC:
                jetCollectionUnskimmed = self.updateJECs(process, jetCollectionUnskimmed, patMetModuleTask, postfix)


        # getting the jet collection that will be used for corrections and uncertainty computation
        # starts with the unskimmed jet collection and applies some selection and cleaning criteria
        jetCollection = self.getJetCollectionForCorsAndUncs(process,
                                                            jetCollectionUnskimmed,
                                                            jetSelection,
                                                            autoJetCleaning,
                                                            patMetModuleTask,
                                                            postfix)

        if onMiniAOD:
            # obtain specific METs (caloMET, DeepMET, PFCHS MET, TRKMET) from MiniAOD
            self.miniAODConfigurationPre(process, patMetModuleTask, pfCandCollection, postfix)
        else:
            from PhysicsTools.PatUtils.pfeGammaToCandidate_cfi import pfeGammaToCandidate
            addToProcessAndTask("pfeGammaToCandidate", pfeGammaToCandidate.clone(
                                  electrons = copy.copy(electronCollection),
                                  photons = copy.copy(photonCollection)),
                                process, patMetModuleTask)
            if hasattr(process,"patElectrons") and process.patElectrons.electronSource == cms.InputTag("reducedEgamma","reducedGedGsfElectrons"):
                process.pfeGammaToCandidate.electron2pf = "reducedEgamma:reducedGsfElectronPfCandMap"
            if hasattr(process,"patPhotons") and process.patPhotons.photonSource == cms.InputTag("reducedEgamma","reducedGedPhotons"):
                process.pfeGammaToCandidate.photon2pf = "reducedEgamma:reducedPhotonPfCandMap"

        # default MET production
        self.produceMET(process, metType, patMetModuleTask, postfix)



        # preparation to run over miniAOD (met reproduction)
        if onMiniAOD:
            self.miniAODConfiguration(process,
                                      pfCandCollection,
                                      jetCollection,
                                      patMetModuleTask,
                                      postfix
                                      )

        ###########################
        ### 3. (re-)correct MET ###
        ###########################

        patMetCorrectionTask = cms.Task()
        metModName = self.getCorrectedMET(process, metType, correctionLevel,
                                                                    produceIntermediateCorrections,
                                                                    jetCollection,
                                                                    patMetCorrectionTask, postfix )
        # fix the default jets for the type1 computation to those used to compute the uncertainties
        # in order to be consistent with what is done in the correction and uncertainty step
        # particularly true for miniAODs
        if "T1" in metModName:
            getattr(process,"patPFMetT1T2Corr"+postfix).src = jetCollection
            getattr(process,"patPFMetT2Corr"+postfix).src = jetCollection
            #ZD:puppi currently doesn't have the L1 corrections in the GT
            if self._parameters["Puppi"].value:
                getattr(process,"patPFMetT1T2Corr"+postfix).offsetCorrLabel = cms.InputTag("")
                getattr(process,"patPFMetT2Corr"+postfix).offsetCorrLabel = cms.InputTag("")
        if "Smear" in metModName:
            getattr(process,"patSmearedJets"+postfix).src = jetCollection
            if self._parameters["Puppi"].value:
                getattr(process,"patPFMetT1T2SmearCorr"+postfix).offsetCorrLabel = cms.InputTag("")


        ####################################
        ### 4. compute MET uncertainties ###
        ####################################

        patMetUncertaintyTask = cms.Task()
        if not hasattr(process, "patMetUncertaintyTask"+postfix):
            if self._parameters["Puppi"].value:
                patMetUncertaintyTask.add(cms.Task(getattr(process, "ak4PFPuppiL1FastL2L3CorrectorTask"),getattr(process, "ak4PFPuppiL1FastL2L3ResidualCorrectorTask")))
            else:
                patMetUncertaintyTask.add(cms.Task(getattr(process, "ak4PFCHSL1FastL2L3CorrectorTask"),getattr(process, "ak4PFCHSL1FastL2L3ResidualCorrectorTask")))
        patShiftedModuleTask = cms.Task()
        if computeUncertainties:
            self.getMETUncertainties(process, metType, metModName,
                                        electronCollection,
                                        photonCollection,
                                        muonCollection,
                                        tauCollection,
                                        pfCandCollection,
                                        jetCollection,
                                        jetUncInfos,
                                        patMetUncertaintyTask,
                                        postfix)

        ####################################
        ### 5. Bring everything together ###
        ####################################

        # add main MET tasks to process
        addTaskToProcess(process, "patMetCorrectionTask"+postfix, patMetCorrectionTask)
        addTaskToProcess(process, "patMetUncertaintyTask"+postfix, patMetUncertaintyTask)
        addTaskToProcess(process, "patShiftedModuleTask"+postfix, patShiftedModuleTask)
        addTaskToProcess(process, "patMetModuleTask"+postfix, patMetModuleTask)

        # prepare and fill the final task containing all the sub-tasks
        fullPatMetTask = cms.Task()
        fullPatMetTask.add(getattr(process, "patMetModuleTask"+postfix))
        fullPatMetTask.add(getattr(process, "patMetCorrectionTask"+postfix))
        fullPatMetTask.add(getattr(process, "patMetUncertaintyTask"+postfix))
        fullPatMetTask.add(getattr(process, "patShiftedModuleTask"+postfix))

        # include calo MET in final MET task
        if hasattr(process, "patCaloMet"):
            fullPatMetTask.add(getattr(process, "patCaloMet"))
        # include deepMETsResolutionTune and deepMETsResponseTune into final MET task
        if hasattr(process, "deepMETsResolutionTune"):
            fullPatMetTask.add(getattr(process, "deepMETsResolutionTune"))
        if hasattr(process, "deepMETsResponseTune"):
            fullPatMetTask.add(getattr(process, "deepMETsResponseTune"))
        # adding the slimmed MET module to final MET task
        if hasattr(process, "slimmedMETs"+postfix):
            fullPatMetTask.add(getattr(process, "slimmedMETs"+postfix))

        # add final MET task to the process
        addTaskToProcess(process, "fullPatMetTask"+postfix, fullPatMetTask)

        # add final MET task to the complete PatAlgosTools task
        task = getPatAlgosToolsTask(process)
        task.add(getattr(process,"fullPatMetTask"+postfix))

        #removing the non used jet selectors
        #configtools.removeIfInSequence(process, "selectedPatJetsForMetT1T2Corr", "patPFMetT1T2CorrSequence", postfix )

        #last modification for miniAODs
        self.miniAODConfigurationPost(process, postfix)

        # insert the fullPatMetSequence into patDefaultSequence if needed
        if addToPatDefaultSequence:
            if not hasattr(process, "patDefaultSequence"):
                raise ValueError("PAT default sequence is not defined !!")
            process.patDefaultSequence += getattr(process, "fullPatMetSequence"+postfix)

#====================================================================================================
    def produceMET(self, process,  metType, patMetModuleTask, postfix):
        # create a local task and a corresponding label to collect all the modules added in this function
        produceMET_task, produceMET_label = cms.Task(), "produceMET_task{}".format(postfix)

        # if PF MET is requested and not already part of the process object, then load the necessary configs and add them to the subtask
        if metType == "PF" and not hasattr(process, 'pat'+metType+'Met'):
            process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")
            produceMET_task.add(process.producePatPFMETCorrectionsTask)
            produceMET_task.add(process.patPFMetT2SmearCorrTask)
            produceMET_task.add(process.patPFMetTxyCorrTask)
            produceMET_task.add(process.jetCorrectorsTask)

        # account for a possible postfix
        _myPatMet = 'pat'+metType+'Met'+postfix
        # if a postfix is requested, the MET type is PF MET, and there is not already an associated object in the process object, then add the needed modules
        if postfix != "" and metType == "PF" and not hasattr(process, _myPatMet):
            noClonesTmp = [ "particleFlowDisplacedVertex", "pfCandidateToVertexAssociation" ]
            # clone the PF MET correction task, add it to the process with a postfix, and add it to the patAlgosToolsTask but exclude the modules above
            # QUESTION: is it possible to add this directly to the subtask?
            configtools.cloneProcessingSnippetTask(process, getattr(process,"producePatPFMETCorrectionsTask"), postfix, noClones = noClonesTmp)
            produceMET_task.add(getattr(process,"producePatPFMETCorrectionsTask"+postfix))
            # add a clone of the patPFMet producer to the process and the subtask
            addToProcessAndTask(_myPatMet,  getattr(process,'patPFMet').clone(), process, produceMET_task)
            # adapt some inputs of the patPFMet producer to account e.g. for the postfix
            getattr(process, _myPatMet).metSource = cms.InputTag("pfMet"+postfix)
            getattr(process, _myPatMet).srcPFCands = copy.copy(self.getvalue("pfCandCollection"))
            # account for possibility of Puppi
            if self.getvalue("Puppi"):
                getattr(process, _myPatMet).srcWeights = self._parameters['puppiProducerForMETLabel'].value
        # set considered electrons, muons, and photons depending on data tier
        if metType == "PF":
            getattr(process, _myPatMet).srcLeptons = \
              cms.VInputTag(copy.copy(self.getvalue("electronCollection")) if self.getvalue("onMiniAOD") else
                              cms.InputTag("pfeGammaToCandidate","electrons"),
                            copy.copy(self.getvalue("muonCollection")),
                            copy.copy(self.getvalue("photonCollection")) if self.getvalue("onMiniAOD") else
                              cms.InputTag("pfeGammaToCandidate","photons"))
        # if running on data, remove generator quantities
        if self.getvalue("runOnData"):
            getattr(process, _myPatMet).addGenMET  = False

        # add PAT MET producer to subtask
        produceMET_task.add(getattr(process, _myPatMet ))

        # add the local task to the process
        addTaskToProcess(process, produceMET_label, produceMET_task)

        # add the task to the patMetModuleTask of the toolCode function
        patMetModuleTask.add(getattr(process, produceMET_label))

#====================================================================================================
    def getCorrectedMET(self, process, metType, correctionLevel, produceIntermediateCorrections,
                        jetCollection, metModuleTask, postfix):

        # default outputs
        getCorrectedMET_task, getCorrectedMET_label = cms.Task(), "getCorrectedMET_task{}".format(postfix)
        # metModName -> metModuleName
        metModName = "pat"+metType+"Met"+postfix

        # names of correction types
        # not really needed but in case we have changes in the future ...
        corTypeNames = {
            "T0":"T0pc",
            "T1":"T1",
            "T2":"T2",
            "Txy":"Txy",
            "Smear":"Smear",
            }


        # if empty correction level, no need to try something or stop if an unknown correction type is used
        for corType in correctionLevel:
            if corType not in corTypeNames.keys():
                if corType != "":
                    raise ValueError(corType+" is not a proper MET correction name! Aborting the MET correction production")
                else:
                    return metModName

        # names of the tasks implementing a specific corretion type, see PatUtils/python/patPFMETCorrections_cff.py
        corTypeTaskNames = {
            "T0": "patPFMetT0CorrTask"+postfix,
            "T1": "patPFMetT1T2CorrTask"+postfix,
            "T2": "patPFMetT2CorrTask"+postfix,
            "Txy": "patPFMetTxyCorrTask"+postfix,
            "Smear": "patPFMetSmearCorrTask"+postfix,
            "T2Smear": "patPFMetT2SmearCorrTask"+postfix
            }

        # if a postfix is requested, clone the needed correction task to the configs and a add a postfix
        # this adds all the correction tasks for all the other METs e.g. PuppiMET due to the postfix
        if postfix != "":
            noClonesTmp = [ "particleFlowDisplacedVertex", "pfCandidateToVertexAssociation" ]
            if not hasattr(process, "patPFMetT0CorrTask"+postfix):
                configtools.cloneProcessingSnippetTask(process, getattr(process,"patPFMetT0CorrTask"), postfix, noClones = noClonesTmp)
                getCorrectedMET_task.add(getattr(process,"patPFMetT0CorrTask"+postfix))
            if not hasattr(process, "patPFMetT1T2CorrTask"+postfix):
                configtools.cloneProcessingSnippetTask(process, getattr(process,"patPFMetT1T2CorrTask"), postfix)
                getCorrectedMET_task.add(getattr(process,"patPFMetT1T2CorrTask"+postfix))
            if not hasattr(process, "patPFMetT2CorrTask"+postfix):
                configtools.cloneProcessingSnippetTask(process, getattr(process,"patPFMetT2CorrTask"), postfix)
                getCorrectedMET_task.add(getattr(process,"patPFMetT2CorrTask"+postfix))
            if not hasattr(process, "patPFMetTxyCorrTask"+postfix):
                configtools.cloneProcessingSnippetTask(process, getattr(process,"patPFMetTxyCorrTask"), postfix)
                getCorrectedMET_task.add(getattr(process,"patPFMetTxyCorrTask"+postfix))
            if not hasattr(process, "patPFMetSmearCorrTask"+postfix):
                configtools.cloneProcessingSnippetTask(process, getattr(process,"patPFMetSmearCorrTask"), postfix)
                getCorrectedMET_task.add(getattr(process,"patPFMetSmearCorrTask"+postfix))
            if not hasattr(process, "patPFMetT2SmearCorrTask"+postfix):
                configtools.cloneProcessingSnippetTask(process, getattr(process,"patPFMetT2SmearCorrTask"), postfix)
                getCorrectedMET_task.add(getattr(process,"patPFMetT2SmearCorrTask"+postfix))

        # collect the MET correction tasks, which have been added to the process, in a dict
        corTypeTasks = {}
        for corType in corTypeTaskNames.keys():
            corTypeTasks[corType] = getattr(process, corTypeTaskNames[corType] )

        # the names of the products which are created by the MET correction tasks and added to the event
        corTypeTags = {
            "T0":['patPFMetT0Corr'+postfix,''],
            "T1":['patPFMetT1T2Corr'+postfix, 'type1'],
            "T2":['patPFMetT2Corr'+postfix,   'type2'],
            "Txy": ['patPFMetTxyCorr'+postfix,''],
            "Smear":['patPFMetT1T2SmearCorr'+postfix, 'type1'],
            "T2Smear":['patPFMetT2SmearCorr'+postfix, 'type2']
            }

        # build the correction string (== correction level), collect the corresponding corrections, and collect the needed tasks
        correctionScheme=""
        correctionProducts = []
        correctionTasks = []
        for corType in correctionLevel:
            correctionScheme += corTypeNames[corType]
            correctionProducts.append(cms.InputTag(corTypeTags[corType][0],corTypeTags[corType][1]))
            correctionTasks.append(corTypeTasks[corType])

        # T2 and smearing corModuleTag switch, specific case
        if "T2" in correctionLevel and "Smear" in correctionLevel:
            correctionProducts.append(cms.InputTag(corTypeTags["T2Smear"][0],corTypeTags["T2Smear"][1]))
            correctionTasks.append(corTypeTasks["T2Smear"])

        # if both are here, consider smeared corJets for the full T1+Smear correction
        if "T1" in correctionLevel and "Smear" in correctionLevel:
            correctionProducts.remove(cms.InputTag(corTypeTags["T1"][0],corTypeTags["T1"][1]))

        # Txy parameter tuning
        if "Txy" in correctionLevel:
            datamc = "DATA" if self.getvalue("runOnData") else "MC"
            self.tuneTxyParameters(process, correctionScheme, postfix, datamc, self.getvalue("campaign"), self.getvalue("era"))
            getattr(process, "patPFMetTxyCorr"+postfix).srcPFlow = self._parameters["pfCandCollection"].value
            if self.getvalue("Puppi"):
                getattr(process, "patPFMetTxyCorr"+postfix).srcWeights = self._parameters['puppiProducerForMETLabel'].value


        # Enable MET significance if the type1 MET is computed
        if "T1" in correctionLevel:
            _myPatMet = "pat"+metType+"Met"+postfix
            getattr(process, _myPatMet).computeMETSignificance = cms.bool(self.getvalue("computeMETSignificance"))
            getattr(process, _myPatMet).srcPFCands = copy.copy(self.getvalue("pfCandCollection"))
            getattr(process, _myPatMet).srcLeptons = \
              cms.VInputTag(copy.copy(self.getvalue("electronCollection")) if self.getvalue("onMiniAOD") else
                              cms.InputTag("pfeGammaToCandidate","electrons"),
                            copy.copy(self.getvalue("muonCollection")),
                            copy.copy(self.getvalue("photonCollection")) if self.getvalue("onMiniAOD") else
                              cms.InputTag("pfeGammaToCandidate","photons"))
            if postfix=="NoHF":
                getattr(process, _myPatMet).computeMETSignificance = cms.bool(False)
            if self.getvalue("runOnData"):
                from RecoMET.METProducers.METSignificanceParams_cfi import METSignificanceParams_Data
                getattr(process, _myPatMet).parameters = METSignificanceParams_Data
            if self.getvalue("Puppi"):
                getattr(process, _myPatMet).srcWeights = self._parameters['puppiProducerForMETLabel'].value
                getattr(process, _myPatMet).srcJets = cms.InputTag('cleanedPatJets'+postfix)
                getattr(process, _myPatMet).srcJetSF = 'AK4PFPuppi'
                getattr(process, _myPatMet).srcJetResPt = 'AK4PFPuppi_pt'
                getattr(process, _myPatMet).srcJetResPhi = 'AK4PFPuppi_phi'

        # MET significance bypass for the patMETs from AOD
        if not self._parameters["onMiniAOD"].value and not postfix=="NoHF":
            _myPatMet = "patMETs"+postfix
            getattr(process, _myPatMet).computeMETSignificance = cms.bool(self.getvalue("computeMETSignificance"))
            getattr(process, _myPatMet).srcPFCands=copy.copy(self.getvalue("pfCandCollection"))
            getattr(process, _myPatMet).srcLeptons = \
              cms.VInputTag(copy.copy(self.getvalue("electronCollection")) if self.getvalue("onMiniAOD") else
                              cms.InputTag("pfeGammaToCandidate","electrons"),
                            copy.copy(self.getvalue("muonCollection")),
                            copy.copy(self.getvalue("photonCollection")) if self.getvalue("onMiniAOD") else
                              cms.InputTag("pfeGammaToCandidate","photons"))
            if self.getvalue("Puppi"):
                getattr(process, _myPatMet).srcWeights = self._parameters['puppiProducerForMETLabel'].value

        if hasattr(process, "patCaloMet"):
            getattr(process, "patCaloMet").computeMETSignificance = cms.bool(False)

        # T1 parameter tuning when CHS jets are not used
        if "T1" in correctionLevel and not self._parameters["CHS"].value and not self._parameters["Puppi"].value:
            addToProcessAndTask("corrPfMetType1"+postfix, getattr(process, "corrPfMetType1" ).clone(), process, getCorrectedMET_task)
            getattr(process, "corrPfMetType1"+postfix).src =  cms.InputTag("ak4PFJets"+postfix)
            getattr(process, "corrPfMetType1"+postfix).jetCorrLabel = cms.InputTag("ak4PFL1FastL2L3Corrector")
            getattr(process, "corrPfMetType1"+postfix).jetCorrLabelRes = cms.InputTag("ak4PFL1FastL2L3ResidualCorrector")
            getattr(process, "corrPfMetType1"+postfix).offsetCorrLabel = cms.InputTag("ak4PFL1FastjetCorrector")
            getattr(process, "basicJetsForMet"+postfix).offsetCorrLabel = cms.InputTag("ak4PFL1FastjetCorrector")

        if "T1" in correctionLevel and self._parameters["Puppi"].value:
            addToProcessAndTask("corrPfMetType1"+postfix, getattr(process, "corrPfMetType1" ).clone(), process, getCorrectedMET_task)
            getattr(process, "corrPfMetType1"+postfix).src =  cms.InputTag("ak4PFJets"+postfix)
            getattr(process, "corrPfMetType1"+postfix).jetCorrLabel = cms.InputTag("ak4PFPuppiL1FastL2L3Corrector")
            getattr(process, "corrPfMetType1"+postfix).jetCorrLabelRes = cms.InputTag("ak4PFPuppiL1FastL2L3ResidualCorrector")
            getattr(process, "corrPfMetType1"+postfix).offsetCorrLabel = cms.InputTag("ak4PFPuppiL1FastjetCorrector")
            getattr(process, "basicJetsForMet"+postfix).offsetCorrLabel = cms.InputTag("L1FastJet")

        if "T1" in correctionLevel and self._parameters["CHS"].value and self._parameters["reclusterJets"].value:
            getattr(process, "corrPfMetType1"+postfix).src =  cms.InputTag("ak4PFJetsCHS"+postfix)

        # create the main MET producer with the applied correction scheme
        metModName = "pat"+metType+"Met"+correctionScheme+postfix

        taskName=""
        corMetProducer=None
        
        # this should always be true due to the way e.g. PuppiMET is handled (metType is set back to PF and the puppi flag is set to true instead)
        if metType == "PF":
            corMetProducer = cms.EDProducer("CorrectedPATMETProducer",
                       src = cms.InputTag('pat'+metType+'Met' + postfix),
                       srcCorrections = cms.VInputTag(correctionProducts)
                     )
            taskName="getCorrectedMET_task"

        addToProcessAndTask(metModName, corMetProducer, process, getCorrectedMET_task)

        # adding the full sequence only if it does not exist
        if not hasattr(process, getCorrectedMET_label):
            for corTask in correctionTasks:
                getCorrectedMET_task.add(corTask)
        # if it exists, only add the missing correction modules, no need to redo everything
        else:
            for corType in corTypeTaskNames.keys():
                if not configtools.contains(getCorrectedMET_task, corTypeTags[corType][0]) and corType in correctionLevel:
                    getCorrectedMET_task.add(corTypeTasks[corType])

        # plug the main patMetproducer
        getCorrectedMET_task.add(getattr(process, metModName))
        
        addTaskToProcess(process, getCorrectedMET_label, getCorrectedMET_task)
        metModuleTask.add(getattr(process, getCorrectedMET_label))

        # create the intermediate MET steps
        # and finally add the met producers in the sequence for scheduled mode
        if produceIntermediateCorrections:
            interMets = self.addIntermediateMETs(process, metType, correctionLevel, correctionScheme, corTypeTags,corTypeNames, postfix)
            for met in interMets.keys():
                addToProcessAndTask(met, interMets[met], process, getCorrectedMET_task)

        return metModName


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
                    corrections.append( cms.InputTag(corTags[ cKey ][0], corTags[ cKey ][1]) )

            if corName == corScheme:
                continue

            corName='pat'+metType+'Met' + corName + postfix
            if configtools.contains(getattr(process,"getCorrectedMET_task"+postfix), corName ) and hasattr(process, corName):
                continue

            interMets[corName] =  cms.EDProducer("CorrectedPATMETProducer",
                 src = cms.InputTag('pat'+metType+'Met' + postfix),
                 srcCorrections = cms.VInputTag(corrections)
               )


        return interMets


#====================================================================================================
    def getMETUncertainties(self, process, metType, metModName, electronCollection,
                            photonCollection, muonCollection, tauCollection,
                            pfCandCollection, jetCollection, jetUncInfos,
                            patMetUncertaintyTask,
                            postfix):
        # create a local task and a corresponding label to collect all the modules added in this function
        getMETUncertainties_task, getMETUncertainties_label = cms.Task(), "getMETUncertainties_task{}".format(postfix)

        #===================================================================================
        # jet energy resolution shifts
        #===================================================================================
        if not isValidInputTag(jetCollection): #or jetCollection=="":
            print("INFO : jet collection %s does not exists, no energy resolution shifting will be performed in MET uncertainty tools" % jetCollection)
        else:
            preId=""
            if "Smear" in metModName:
                preId="Smeared"

            metJERUncModules = self.getVariations(process, metModName, "Jet",preId, jetCollection, "Res", patMetUncertaintyTask, postfix=postfix )

            for mod in metJERUncModules.keys():
                addToProcessAndTask(mod, metJERUncModules[mod], process, getMETUncertainties_task)

        #===================================================================================
        # Unclustered energy candidates
        #===================================================================================
        if not hasattr(process, "pfCandsForUnclusteredUnc"+postfix):

            #Jet projection ==
            pfCandsNoJets = _mod.candPtrProjector.clone( 
                                           src = pfCandCollection,
                                           veto = jetCollection,
                                           )
            addToProcessAndTask("pfCandsNoJets"+postfix, pfCandsNoJets, process, getMETUncertainties_task)

            #electron projection ==
            pfCandsNoJetsNoEle = _mod.candPtrProjector.clone(
                                                src = "pfCandsNoJets"+postfix,
                                                veto = electronCollection,
                                                )
            if not self.getvalue("onMiniAOD"):
              pfCandsNoJetsNoEle.veto = "pfeGammaToCandidate:electrons"
            addToProcessAndTask("pfCandsNoJetsNoEle"+postfix, pfCandsNoJetsNoEle, process, getMETUncertainties_task)

            #muon projection ==
            pfCandsNoJetsNoEleNoMu = _mod.candPtrProjector.clone(
                                              src = "pfCandsNoJetsNoEle"+postfix,
                                              veto = muonCollection,
                                              )
            addToProcessAndTask("pfCandsNoJetsNoEleNoMu"+postfix, pfCandsNoJetsNoEleNoMu, process, getMETUncertainties_task)

            #tau projection ==
            pfCandsNoJetsNoEleNoMuNoTau = _mod.candPtrProjector.clone(
                                              src = "pfCandsNoJetsNoEleNoMu"+postfix,
                                              veto = tauCollection,
                                              )
            addToProcessAndTask("pfCandsNoJetsNoEleNoMuNoTau"+postfix, pfCandsNoJetsNoEleNoMuNoTau, process, getMETUncertainties_task)

            #photon projection ==
            pfCandsForUnclusteredUnc = _mod.candPtrProjector.clone(
                                              src = "pfCandsNoJetsNoEleNoMuNoTau"+postfix,
                                              veto = photonCollection,
                                              )
            if not self.getvalue("onMiniAOD"):
              pfCandsForUnclusteredUnc.veto = "pfeGammaToCandidate:photons"
            addToProcessAndTask("pfCandsForUnclusteredUnc"+postfix, pfCandsForUnclusteredUnc, process, getMETUncertainties_task)

        #===================================================================================
        # energy shifts
        #===================================================================================
        # PFMuons, PFElectrons, PFPhotons, and PFTaus will be used
        # to calculate MET Uncertainties.
        #===================================================================================
        #--------------
        # PFElectrons :
        #--------------
        pfElectrons = cms.EDFilter("CandPtrSelector",
                                   src = electronCollection,
                                   cut = cms.string("pt > 5 && isPF && gsfTrack.isAvailable() && gsfTrack.hitPattern().numberOfLostHits(\'MISSING_INNER_HITS\') < 2")
                                   )
        addToProcessAndTask("pfElectrons"+postfix, pfElectrons, process, getMETUncertainties_task)
        #--------------------------------------------------------------------
        # PFTaus :
        #---------
        pfTaus = cms.EDFilter("PATTauRefSelector",
                              src = tauCollection,
                              cut = cms.string('pt > 18.0 & abs(eta) < 2.6 & tauID("decayModeFinding") > 0.5 & isPFTau')
                              )
        addToProcessAndTask("pfTaus"+postfix, pfTaus, process, getMETUncertainties_task)
        #---------------------------------------------------------------------
        # PFMuons :
        #----------
        pfMuons = cms.EDFilter("CandPtrSelector",
                               src = muonCollection,
                               cut = cms.string("pt > 5.0 && isPFMuon && abs(eta) < 2.4")
                               )
        addToProcessAndTask("pfMuons"+postfix, pfMuons, process, getMETUncertainties_task)
        #---------------------------------------------------------------------
        # PFPhotons :
        #------------
        if self._parameters["Puppi"].value or not self._parameters["onMiniAOD"].value:
            cutforpfNoPileUp = cms.string("")
        else:
            cutforpfNoPileUp = cms.string("fromPV > 1")

        pfNoPileUp = cms.EDFilter("CandPtrSelector",
                                  src = pfCandCollection,
                                  cut = cutforpfNoPileUp
                                  )
        addToProcessAndTask("pfNoPileUp"+postfix, pfNoPileUp, process, getMETUncertainties_task)

        pfPhotons = cms.EDFilter("CandPtrSelector",
                                 src = pfCandCollection if self._parameters["Puppi"].value or not self._parameters["onMiniAOD"].value else cms.InputTag("pfCHS"),
                                 cut = cms.string("abs(pdgId) = 22")
                                 )
        addToProcessAndTask("pfPhotons"+postfix, pfPhotons, process, getMETUncertainties_task)
        #-------------------------------------------------------------------------
        # Collections which have only PF Objects for calculating MET uncertainties
        #-------------------------------------------------------------------------
        electronCollection = cms.InputTag("pfElectrons"+postfix)
        muonCollection     = cms.InputTag("pfMuons"+postfix)
        tauCollection      = cms.InputTag("pfTaus"+postfix)
        photonCollection   = cms.InputTag("pfPhotons"+postfix)


        objectCollections = { "Jet":jetCollection,
                              "Electron":electronCollection,
                              "Photon":photonCollection,
                              "Muon":muonCollection,
                              "Unclustered":cms.InputTag("pfCandsForUnclusteredUnc"+postfix),
                              "Tau":tauCollection,
                              }

        for obj in objectCollections.keys():
            if not isValidInputTag(objectCollections[obj]): # or objectCollections[obj]=="":
                print("INFO : %s collection %s does not exists, no energy scale shifting will be performed in MET uncertainty tools" %(obj, objectCollections[obj]))
            else:
                metObjUncModules = self.getVariations(process, metModName, obj,"", objectCollections[obj], "En", patMetUncertaintyTask, jetUncInfos, postfix )

                #adding the shifted MET produced to the proper patMetModuleSequence
                for mod in metObjUncModules.keys():
                    addToProcessAndTask(mod, metObjUncModules[mod], process, getMETUncertainties_task)

        # add the local task to the process
        addTaskToProcess(process, getMETUncertainties_label, getMETUncertainties_task)
        patMetUncertaintyTask.add(getattr(process, getMETUncertainties_label))

#====================================================================================================
    def createEnergyScaleShiftedUpModule(self, process,identifier, objectCollection,
                                         varyByNsigmas, jetUncInfos=None, postfix=""):

        shiftedModuleUp = None

        if identifier == "Electron":
            shiftedModuleUp = cms.EDProducer("ShiftedParticleProducer",
                                             src = objectCollection,
                                             uncertainty = cms.string('((abs(y)<1.479)?(0.006+0*x):(0.015+0*x))'),
                                             shiftBy = cms.double(+1.*varyByNsigmas),
                                             srcWeights = cms.InputTag("")
                                             )

        if identifier == "Photon":
            shiftedModuleUp = cms.EDProducer("ShiftedParticleProducer",
                                             src = objectCollection,
                                             uncertainty = cms.string('((abs(y)<1.479)?(0.01+0*x):(0.025+0*x))'),
                                             shiftBy = cms.double(+1.*varyByNsigmas),
                                             srcWeights = cms.InputTag("")
                                             )

        if identifier == "Muon":
            shiftedModuleUp = cms.EDProducer("ShiftedParticleProducer",
                                             src = objectCollection,
                                             uncertainty = cms.string('((x<100)?(0.002+0*y):(0.05+0*y))'),
                                             shiftBy = cms.double(+1.*varyByNsigmas),
                                             srcWeights = cms.InputTag("")
                                             )

        if identifier == "Tau":
            shiftedModuleUp = cms.EDProducer("ShiftedParticleProducer",
                                             src = objectCollection,
                                             uncertainty = cms.string('0.03+0*x*y'),
                                             shiftBy = cms.double(+1.*varyByNsigmas),
                                             srcWeights = cms.InputTag("")
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
                                             shiftBy = cms.double(+1.*varyByNsigmas),
                                             srcWeights = cms.InputTag("")
                                             )

        if identifier == "Jet":
            moduleType="ShiftedPATJetProducer"

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
    def tuneTxyParameters(self, process, corScheme, postfix, datamc="", campaign="", era="" ):
        corSchemes = ["Txy", "T1Txy", "T0pcTxy", "T0pcT1Txy", "T1T2Txy", "T0pcT1T2Txy", "T1SmearTxy", "T1T2SmearTxy", "T0pcT1SmearTxy", "T0pcT1T2SmearTxy"]
        import PhysicsTools.PatUtils.patPFMETCorrections_cff as metCors
        xyTags = {}
        for corScheme_ in corSchemes:
            xyTags["{}_{}".format(corScheme_,"50ns")]=getattr(metCors,"{}_{}_{}".format("patMultPhiCorrParams",corScheme_,"50ns"))
            xyTags["{}_{}".format(corScheme_,"25ns")]=getattr(metCors,"{}_{}_{}".format("patMultPhiCorrParams",corScheme_,"25ns"))
            if datamc!="" and campaign!="" and era!="":
                if not self.getvalue("Puppi"):
                    xyTags["{}_{}_{}_{}".format(corScheme_,campaign,datamc,era)]=getattr(metCors,"{}_{}{}{}".format("patMultPhiCorrParams",campaign,datamc,era))
                else:
                    xyTags["{}_{}_{}_{}".format(corScheme_,campaign,datamc,era)]=getattr(metCors,"{}_{}{}{}".format("patMultPhiCorrParams_Puppi",campaign,datamc,era))

        if datamc!="" and campaign!="" and era!="":
            getattr(process, "patPFMetTxyCorr"+postfix).parameters = xyTags["{}_{}_{}_{}".format(corScheme,campaign,datamc,era)]
        else:
            getattr(process, "patPFMetTxyCorr"+postfix).parameters = xyTags[corScheme+"_25ns"]


#====================================================================================================
    def getVariations(self, process, metModName, identifier,preId, objectCollection, varType,
                      patMetUncertaintyTask, jetUncInfos=None, postfix="" ):

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
            else:
                smear=True
                varyByNsigmas=101

            shiftedCollModules['Up'] = self.createShiftedJetResModule(process, smear, objectCollection, +1.*varyByNsigmas,
                                                                      "Up", postfix)
            shiftedCollModules['Down'] = self.createShiftedJetResModule(process, smear, objectCollection, -1.*varyByNsigmas,
                                                                        "Down", postfix)
        else:
            shiftedCollModules['Up'] = self.createEnergyScaleShiftedUpModule(process, identifier, objectCollection, varyByNsigmas, jetUncInfos, postfix)
            shiftedCollModules['Down'] = shiftedCollModules['Up'].clone( shiftBy = cms.double(-1.*varyByNsigmas) )

        if identifier=="Jet" and varType=="Res":
            smear=False
            if "Smear" in metModName:
                objectCollection=cms.InputTag("selectedPatJetsForMetT1T2SmearCorr"+postfix)



        #and the MET producers
        if identifier=="Jet" and varType=="Res" and self._parameters["runOnData"].value:
            shiftedMetProducers = self.copyCentralMETProducer(process, shiftedCollModules, identifier, metModName, varType, postfix)
        else:
            shiftedMetProducers = self.createShiftedModules(process, shiftedCollModules, identifier, preId, objectCollection,
                                                            metModName, varType, patMetUncertaintyTask, postfix)

        return shiftedMetProducers


#========================================================================================
    def copyCentralMETProducer(self, process, shiftedCollModules, identifier, metModName, varType, postfix):

        # remove the postfix to put it at the end
        shiftedMetProducers = {}
        baseName = self.removePostfix(metModName, postfix)
        for mod in shiftedCollModules.keys():
            modName = baseName+identifier+varType+mod+postfix
            shiftedMETModule = getattr(process, metModName).clone()
            shiftedMetProducers[ modName ] = shiftedMETModule

        return shiftedMetProducers


#========================================================================================
    def createShiftedJetResModule(self, process, smear, objectCollection, varyByNsigmas, varDir, postfix ):

        smearedJetModule = self.createSmearedJetModule(process, objectCollection, smear, varyByNsigmas, varDir, postfix)

        return smearedJetModule


#========================================================================================
    def createShiftedModules(self, process, shiftedCollModules, identifier, preId, objectCollection,
                             metModName, varType, patMetUncertaintyTask, postfix):
        # create a local task and a corresponding label to collect all the modules added in this function
        createShiftedModules_task, createShiftedModules_label = cms.Task(), "createShiftedModules_task{}".format(postfix)

        shiftedMetProducers = {}

        # remove the postfix to put it at the end
        baseName = self.removePostfix(metModName, postfix)

        #adding the shifted collection producers to the sequence, create the shifted MET correction Modules and add them as well
        for mod in shiftedCollModules.keys():
            modName = "shiftedPat"+preId+identifier+varType+mod+postfix
            if (identifier=="Photon" or identifier=="Unclustered") and self.getvalue("Puppi"):
                shiftedCollModules[mod].srcWeights = self._parameters['puppiProducerForMETLabel'].value
            if not hasattr(process, modName):
                addToProcessAndTask(modName, shiftedCollModules[mod], process, createShiftedModules_task)

            #removing the uncorrected
            modName = "shiftedPat"+preId+identifier+varType+mod+postfix

            #PF MET =================================================================================
            if "PF" in metModName:
                #create the MET shifts and add them to the sequence
                shiftedMETCorrModule = self.createShiftedMETModule(process, objectCollection, modName)
                if (identifier=="Photon" or identifier=="Unclustered") and self.getvalue("Puppi"):
                   shiftedMETCorrModule.srcWeights = self._parameters['puppiProducerForMETLabel'].value
                modMETShiftName = "shiftedPatMETCorr"+preId+identifier+varType+mod+postfix
                if not hasattr(process, modMETShiftName):
                    addToProcessAndTask(modMETShiftName, shiftedMETCorrModule, process, createShiftedModules_task)

                #and finally prepare the shifted MET producers
                modName = baseName+identifier+varType+mod+postfix
                shiftedMETModule = getattr(process, metModName).clone(
                    src = cms.InputTag( metModName ),
                    srcCorrections = cms.VInputTag( cms.InputTag(modMETShiftName) )
                    )
                shiftedMetProducers[ modName ] = shiftedMETModule

           #==========================================================================================

        addTaskToProcess(process, createShiftedModules_label, createShiftedModules_task)
        patMetUncertaintyTask.add(getattr(process, createShiftedModules_label))

        return shiftedMetProducers


#========================================================================================
    def createShiftedMETModule(self, process, originCollection, shiftedCollection):

        shiftedModule = cms.EDProducer("ShiftedParticleMETcorrInputProducer",
                                       srcOriginal = originCollection,
                                       srcShifted = cms.InputTag(shiftedCollection),
                                       srcWeights = cms.InputTag("")
                                       )

        return shiftedModule

#========================================================================================
    def createSmearedJetModule(self, process, jetCollection, smear, varyByNsigmas, varDir, postfix):

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

        genJetsCollection=cms.InputTag('ak4GenJetsNoNu')
        if self._parameters["onMiniAOD"].value:
            genJetsCollection=cms.InputTag("slimmedGenJets")

        if self._parameters["Puppi"].value:
            getattr(process, "patSmearedJets"+postfix).algo = 'AK4PFPuppi'
            getattr(process, "patSmearedJets"+postfix).algopt = 'AK4PFPuppi_pt'

        if "PF" == self._parameters["metType"].value:
            smearedJetModule = getattr(process, "patSmearedJets"+postfix).clone(
                src = jetCollection,
                enabled = cms.bool(smear),
                variation = cms.int32( int(varyByNsigmas) ),
                genJets = genJetsCollection,
                )

        return smearedJetModule


### Utilities ====================================================================
    def labelsInSequence(process, sequenceLabel, postfix=""):
        result = [ m.label()[:-len(postfix)] for m in listModules( getattr(process,sequenceLabel+postfix))]
        result.extend([ m.label()[:-len(postfix)] for m in listSequences( getattr(process,sequenceLabel+postfix))]  )
        if postfix == "":
            result = [ m.label() for m in listModules( getattr(process,sequenceLabel+postfix))]
            result.extend([ m.label() for m in listSequences( getattr(process,sequenceLabel+postfix))]  )
        return result

    def initializeInputTag(self, input, default):
        retVal = None
        if input is None:
            retVal = self._defaultParameters[default].value
        elif isinstance(input, str):
            retVal = cms.InputTag(input)
        else:
            retVal = input
        return retVal


    def recomputeRawMetFromPfcs(self, process, pfCandCollection, onMiniAOD, patMetModuleTask, postfix):
        # create a local task and a corresponding label to collect all the modules added in this function
        recomputeRawMetFromPfcs_task, recomputeRawMetFromPfcs_label = cms.Task(), "recomputeRawMetFromPfcs_task{}".format(postfix)

        # RECO MET
        if not hasattr(process, "pfMet"+postfix) and self._parameters["metType"].value == "PF":
            # common to AOD/mAOD processing
            # raw MET
            # clone the standard pfMet module, add it to the relevant task, and adapt the input collection to the desired PF candidate collection
            from RecoMET.METProducers.pfMet_cfi import pfMet
            addToProcessAndTask("pfMet"+postfix, pfMet.clone(), process, recomputeRawMetFromPfcs_task)
            getattr(process, "pfMet"+postfix).src = pfCandCollection
            getattr(process, "pfMet"+postfix).calculateSignificance = False
            # if Puppi MET is requested, apply the Puppi weights
            if self.getvalue("Puppi"):
                getattr(process, "pfMet"+postfix).applyWeight = True
                getattr(process, "pfMet"+postfix).srcWeights = self._parameters['puppiProducerForMETLabel'].value

        # PAT MET
        if not hasattr(process, "patMETs"+postfix) and self._parameters["metType"].value == "PF":
            process.load("PhysicsTools.PatAlgos.producersLayer1.metProducer_cff")
            recomputeRawMetFromPfcs_task.add(process.makePatMETsTask)
            configtools.cloneProcessingSnippetTask(process, getattr(process,"patMETCorrectionsTask"), postfix)
            recomputeRawMetFromPfcs_task.add(getattr(process,"patMETCorrectionsTask"+postfix))

            # T1 pfMet for AOD to mAOD only
            if not onMiniAOD: #or self._parameters["Puppi"].value:
                #correction duplication needed
                getattr(process, "pfMetT1"+postfix).src = cms.InputTag("pfMet"+postfix)
                recomputeRawMetFromPfcs_task.add(getattr(process, "pfMetT1"+postfix))
                _myPatMet = 'patMETs'+postfix
                addToProcessAndTask(_myPatMet, getattr(process,'patMETs' ).clone(), process, recomputeRawMetFromPfcs_task)
                getattr(process, _myPatMet).metSource = cms.InputTag("pfMetT1"+postfix)
                getattr(process, _myPatMet).computeMETSignificance = cms.bool(self.getvalue("computeMETSignificance"))
                getattr(process, _myPatMet).srcLeptons = \
                  cms.VInputTag(copy.copy(self.getvalue("electronCollection")) if self.getvalue("onMiniAOD") else
                                  cms.InputTag("pfeGammaToCandidate","electrons"),
                                copy.copy(self.getvalue("muonCollection")),
                                copy.copy(self.getvalue("photonCollection")) if self.getvalue("onMiniAOD") else
                                  cms.InputTag("pfeGammaToCandidate","photons"))
                if postfix=="NoHF":
                    getattr(process, _myPatMet).computeMETSignificance = cms.bool(False)

                if self.getvalue("Puppi"):
                    getattr(process, _myPatMet).srcWeights = self._parameters['puppiProducerForMETLabel'].value
                    getattr(process, _myPatMet).srcJets = cms.InputTag('cleanedPatJets'+postfix)
                    getattr(process, _myPatMet).srcJetSF = cms.string('AK4PFPuppi')
                    getattr(process, _myPatMet).srcJetResPt = cms.string('AK4PFPuppi_pt')
                    getattr(process, _myPatMet).srcJetResPhi = cms.string('AK4PFPuppi_phi')

        # add the local task to the process
        addTaskToProcess(process, recomputeRawMetFromPfcs_label, recomputeRawMetFromPfcs_task)

        # add the task to the patMetModuleTask of the toolCode function
        patMetModuleTask.add(getattr(process, recomputeRawMetFromPfcs_label))

    # function to extract specific MET from the slimmedMETs(Puppi) collection in MiniAOD and then put it into the event as a new reco::MET collection
    def extractMET(self, process, correctionLevel, patMetModuleTask, postfix):
        # create a local task and a corresponding label to collect all the modules added in this function
        extractMET_task, extractMET_label = cms.Task(), "extractMET_task{}".format(postfix)

        # module to extract a MET from the slimmedMETs(Puppi) collection
        pfMet = cms.EDProducer("RecoMETExtractor",
                               metSource= cms.InputTag("slimmedMETs" if not self._parameters["Puppi"].value else "slimmedMETsPuppi",processName=cms.InputTag.skipCurrentProcess()),
                               correctionLevel = cms.string(correctionLevel)
                               )
        if(correctionLevel=="raw"):
            addToProcessAndTask("pfMet"+postfix, pfMet, process, extractMET_task)
        else:
            addToProcessAndTask("met"+correctionLevel+postfix, pfMet, process, extractMET_task)

        if not hasattr(process, "genMetExtractor"+postfix) and not self._parameters["runOnData"].value:
            genMetExtractor = cms.EDProducer("GenMETExtractor",
                                             metSource= cms.InputTag("slimmedMETs",processName=cms.InputTag.skipCurrentProcess())
                                             )
            addToProcessAndTask("genMetExtractor"+postfix, genMetExtractor, process, extractMET_task)

        # add the local task to the process
        addTaskToProcess(process, extractMET_label, extractMET_task)

        # add the task to the patMetModuleTask of the toolCode function
        patMetModuleTask.add(getattr(process, extractMET_label))


    def updateJECs(self,process,jetCollection, patMetModuleTask, postfix):
        from PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cff import updatedPatJetCorrFactors

        patJetCorrFactorsReapplyJEC = updatedPatJetCorrFactors.clone(
            src = jetCollection if not self._parameters["Puppi"].value else cms.InputTag("slimmedJetsPuppi"),
            levels = ['L1FastJet',
                      'L2Relative',
                      'L3Absolute'],
            payload = 'AK4PFchs' if not self._parameters["Puppi"].value else 'AK4PFPuppi' ) # always CHS from miniAODs, except for puppi

        if self._parameters["runOnData"].value:
            patJetCorrFactorsReapplyJEC.levels.append("L2L3Residual")

        from PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cff import updatedPatJets
        patJetsReapplyJEC = updatedPatJets.clone(
            jetSource = jetCollection if not self._parameters["Puppi"].value else cms.InputTag("slimmedJetsPuppi"),
            jetCorrFactorsSource = cms.VInputTag(cms.InputTag("patJetCorrFactorsReapplyJEC"+postfix))
            )

        # create a local task and a corresponding label to collect all the modules added in this function
        updateJECs_task, updateJECs_label = cms.Task(), "updateJECs_task{}".format(postfix)
        addToProcessAndTask("patJetCorrFactorsReapplyJEC"+postfix, patJetCorrFactorsReapplyJEC, process, updateJECs_task)
        addToProcessAndTask("patJetsReapplyJEC"+postfix, patJetsReapplyJEC.clone(), process, updateJECs_task)

        # add the local task to the process
        addTaskToProcess(process, updateJECs_label, updateJECs_task)

        # add the task to the patMetModuleTask of the toolCode function
        patMetModuleTask.add(getattr(process, updateJECs_label))

        return  cms.InputTag("patJetsReapplyJEC"+postfix)


    def getJetCollectionForCorsAndUncs(self, process, jetCollectionUnskimmed,
                                       jetSelection, autoJetCleaning,patMetModuleTask, postfix):
        # create a local task and a corresponding label to collect all the modules added in this function
        getJetCollectionForCorsAndUncs_task, getJetCollectionForCorsAndUncs_label = cms.Task(), "getJetCollectionForCorsAndUncs_task{}".format(postfix)

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
        addToProcessAndTask("basicJetsForMet"+postfix, basicJetsForMet, process, getJetCollectionForCorsAndUncs_task)

        from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets
        jetSelector = selectedPatJets.clone(
            src = cms.InputTag("basicJetsForMet"+postfix),
            cut = cms.string(jetSelection)
            )
        addToProcessAndTask("jetSelectorForMet"+postfix, jetSelector, process, getJetCollectionForCorsAndUncs_task)

        jetCollection = self.jetCleaning(process, "jetSelectorForMet"+postfix, autoJetCleaning, patMetModuleTask, postfix)
        addTaskToProcess(process, getJetCollectionForCorsAndUncs_label, getJetCollectionForCorsAndUncs_task)
        patMetModuleTask.add(getattr(process, getJetCollectionForCorsAndUncs_label))

        return jetCollection


    def ak4JetReclustering(self,process, pfCandCollection, patMetModuleTask, postfix):
        # create a local task and a corresponding label to collect all the modules added in this function
        ak4JetReclustering_task, ak4JetReclustering_label = cms.Task(), "ak4JetReclustering_task{}".format(postfix)

        chs = self._parameters["CHS"].value
        jetColName="ak4PFJets"
        CHSname=""
        pfCandColl=pfCandCollection
        if chs:
            CHSname="chs"
            jetColName="ak4PFJetsCHS"
            if self._parameters["onMiniAOD"].value: 
                pfCandColl = cms.InputTag("pfCHS")
            else:
                addToProcessAndTask("tmpPFCandCollPtr"+postfix,
                                    cms.EDProducer("PFCandidateFwdPtrProducer",
                                                   src = pfCandCollection ),
                                    process, ak4JetReclustering_task)
                process.load("CommonTools.ParticleFlow.pfNoPileUpJME_cff")
                ak4JetReclustering_task.add(process.pfNoPileUpJMETask)
                configtools.cloneProcessingSnippetTask(process, getattr(process,"pfNoPileUpJMETask"), postfix)
                ak4JetReclustering_task.add(getattr(process,"pfNoPileUpJMETask"+postfix))
                getattr(process, "pfPileUpJME"+postfix).PFCandidates = "tmpPFCandCollPtr"+postfix
                getattr(process, "pfNoPileUpJME"+postfix).bottomCollection = "tmpPFCandCollPtr"+postfix
                pfCandColl = "pfNoPileUpJME"+postfix

        jetColName+=postfix
        if not hasattr(process, jetColName):
            from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
            #if chs:
            addToProcessAndTask(jetColName, ak4PFJets.clone(), process, ak4JetReclustering_task)
            getattr(process, jetColName).src = pfCandColl
            getattr(process, jetColName).doAreaFastjet = True

            #puppi
            if self._parameters["Puppi"].value:
                getattr(process, jetColName).srcWeights = cms.InputTag(self._parameters['puppiProducerLabel'].value)
                getattr(process, jetColName).applyWeight = True

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
            getattr(process,"patJets"+postfix).embedGenPartonMatch = False
            getattr(process,"patJets"+postfix).embedGenJetMatch = False
            if self._parameters['onMiniAOD'].value:
                del getattr(process,"patJets"+postfix).JetFlavourInfoSource
                del getattr(process,"patJets"+postfix).JetPartonMapSource
                del getattr(process,"patJets"+postfix).genPartonMatch
                del getattr(process,"patJets"+postfix).genJetMatch
            getattr(process,"patJets"+postfix).getJetMCFlavour = False

            getattr(process,"patJetCorrFactors"+postfix).src=cms.InputTag(jetColName)
            getattr(process,"patJetCorrFactors"+postfix).primaryVertices= cms.InputTag("offlineSlimmedPrimaryVertices")
            if self._parameters["Puppi"].value:
                getattr(process,"patJetCorrFactors"+postfix).payload=cms.string('AK4PFPuppi')

        # add the local task to the process
        addTaskToProcess(process, ak4JetReclustering_label, ak4JetReclustering_task)

        # add the task to the patMetModuleTask of the toolCode function
        patMetModuleTask.add(getattr(process, ak4JetReclustering_label))

        return cms.InputTag("patJets"+postfix)


    # function to add different METs to event starting from slimmedMET collections in MiniAOD
    def miniAODConfigurationPre(self, process, patMetModuleTask, pfCandCollection, postfix):
        # create a local task and a corresponding label to collect all the modules added in this function
        miniAODConfigurationPre_task, miniAODConfigurationPre_label = cms.Task(), "miniAODConfigurationPre_task{}".format(postfix)

        # extract caloMET from MiniAOD
        # "rawCalo" -> hardcoded in PhysicsTools/PatAlgos/plugins/RecoMETExtractor.cc
        self.extractMET(process, "rawCalo", miniAODConfigurationPre_task, postfix)
        caloMetName="metrawCalo" if hasattr(process,"metrawCalo") else "metrawCalo"+postfix
        from PhysicsTools.PatAlgos.tools.metTools import addMETCollection
        addMETCollection(process,
                         labelName = "patCaloMet",
                         metSource = caloMetName
                         )
        getattr(process,"patCaloMet").addGenMET = False
        miniAODConfigurationPre_task.add(getattr(process,"patCaloMet"))

        # extract DeepMETs (ResponseTune and ResolutionTune) from MiniAOD
        # "rawDeepResponseTune" and "rawDeepResolutionTune" hardcoded in PhysicsTools/PatAlgos/plugins/RecoMETExtractor.cc
        if self._parameters["extractDeepMETs"].value:
            self.extractMET(process, "rawDeepResponseTune", miniAODConfigurationPre_task, postfix)
            deepMetResponseTuneName = "metrawDeepResponseTune" if hasattr(process, "metrawDeepResponseTune") else "metrawDeepResponseTune"+postfix
            addMETCollection(process,
                             labelName = "deepMETsResponseTune",
                             metSource = deepMetResponseTuneName
                            )
            getattr(process, "deepMETsResponseTune").addGenMET = False
            getattr(process, "deepMETsResponseTune").computeMETSignificance = cms.bool(False)
            miniAODConfigurationPre_task.add(getattr(process, "deepMETsResponseTune"))

            self.extractMET(process, "rawDeepResolutionTune", miniAODConfigurationPre_task, postfix)
            deepMetResolutionTuneName = "metrawDeepResolutionTune" if hasattr(process, "metrawDeepResolutionTune") else "metrawDeepResolutionTune"+postfix
            addMETCollection(process,
                             labelName = "deepMETsResolutionTune",
                             metSource = deepMetResolutionTuneName
                            )
            getattr(process, "deepMETsResolutionTune").addGenMET = False
            getattr(process, "deepMETsResolutionTune").computeMETSignificance = cms.bool(False)
            miniAODConfigurationPre_task.add(getattr(process, "deepMETsResolutionTune"))


        # obtain PFCHS MET and TRK MET
        # adding the necessary chs and track met configuration

        from CommonTools.ParticleFlow.pfCHS_cff import pfCHS
        addToProcessAndTask("pfCHS", pfCHS.clone(), process, miniAODConfigurationPre_task)
        from RecoMET.METProducers.pfMet_cfi import pfMet
        pfMetCHS = pfMet.clone(src = "pfCHS")
        addToProcessAndTask("pfMetCHS", pfMetCHS, process, miniAODConfigurationPre_task)

        addMETCollection(process,
                         labelName = "patCHSMet",
                         metSource = "pfMetCHS"
                         )

        process.patCHSMet.computeMETSignificant = cms.bool(False)
        process.patCHSMet.addGenMET = cms.bool(False)
        miniAODConfigurationPre_task.add(process.patCHSMet)

        # get pf met only considering charged particles matched to primary vertex -> TRK MET
        pfTrk = chargedPackedCandsForTkMet.clone()
        addToProcessAndTask("pfTrk", pfTrk, process, miniAODConfigurationPre_task)
        pfMetTrk = pfMet.clone(src = 'pfTrk')
        addToProcessAndTask("pfMetTrk", pfMetTrk, process, miniAODConfigurationPre_task)

        addMETCollection(process,
                         labelName = "patTrkMet",
                         metSource = "pfMetTrk"
                         )

        process.patTrkMet.computeMETSignificant = cms.bool(False)
        process.patTrkMet.addGenMET = cms.bool(False)
        miniAODConfigurationPre_task.add(process.patTrkMet)

        addTaskToProcess(process, miniAODConfigurationPre_label, miniAODConfigurationPre_task)

        patMetModuleTask.add(getattr(process, miniAODConfigurationPre_label))

    def miniAODConfigurationPost(self, process, postfix):

        if self._parameters["metType"].value == "PF":
            if hasattr(process, "patPFMetTxyCorr"+postfix):
                getattr(process, "patPFMetTxyCorr"+postfix).vertexCollection = cms.InputTag("offlineSlimmedPrimaryVertices")

        if self._parameters['computeUncertainties'].value and not self._parameters["runOnData"]:
            getattr(process, "shiftedPatJetResDown"+postfix).genJets = cms.InputTag("slimmedGenJets")
            getattr(process, "shiftedPatJetResUp"+postfix).genJets = cms.InputTag("slimmedGenJets")


    def miniAODConfiguration(self, process, pfCandCollection, jetCollection,
                             patMetModuleTask, postfix ):
        # specify options for regular pat PFMET
        if self._parameters["metType"].value == "PF": # not hasattr(process, "pfMet"+postfix)
            if "T1" in self._parameters['correctionLevel'].value:
                getattr(process, "patPFMet"+postfix).srcJets = jetCollection
            getattr(process, "patPFMet"+postfix).addGenMET  = False
            if not self._parameters["runOnData"].value:
                getattr(process, "patPFMet"+postfix).addGenMET  = True
                getattr(process, "patPFMet"+postfix).genMETSource = cms.InputTag("genMetExtractor"+postfix)

        # if smeared PFMET is needed, use the ak4 gen jets saved in MiniAOD
        if "Smear" in self._parameters['correctionLevel'].value:
            getattr(process, "patSmearedJets"+postfix).genJets = cms.InputTag("slimmedGenJets")

        # add a new slimmedMETs collection and configure it properly
        if not hasattr(process, "slimmedMETs"+postfix) and self._parameters["metType"].value == "PF":

            # create a local task and a corresponding label to collect all the modules added in this function
            miniAODConfiguration_task, miniAODConfiguration_label = cms.Task(), "miniAODConfiguration_task{}".format(postfix)

            from PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi import slimmedMETs
            addToProcessAndTask("slimmedMETs"+postfix, slimmedMETs.clone(), process, miniAODConfiguration_task)
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
            getattr(process,"slimmedMETs"+postfix).t01Variation = cms.InputTag("slimmedMETs" if not self._parameters["Puppi"].value else "slimmedMETsPuppi",processName=cms.InputTag.skipCurrentProcess())

            if hasattr(process, "deepMETsResolutionTune") and hasattr(process, "deepMETsResponseTune"):
                # process includes producing/extracting deepMETsResolutionTune and deepMETsResponseTune
                # add them to the slimmedMETs
                getattr(process,"slimmedMETs"+postfix).addDeepMETs = True

            # smearing and type0 variations not yet supported in reprocessing
            #del getattr(process,"slimmedMETs"+postfix).t1SmearedVarsAndUncs
            del getattr(process,"slimmedMETs"+postfix).tXYUncForRaw
            del getattr(process,"slimmedMETs"+postfix).tXYUncForT01
            del getattr(process,"slimmedMETs"+postfix).tXYUncForT1Smear
            del getattr(process,"slimmedMETs"+postfix).tXYUncForT01Smear
            #del getattr(process,"slimmedMETs"+postfix).caloMET

            # add the local task to the process
            addTaskToProcess(process, miniAODConfiguration_label, miniAODConfiguration_task)

            # add the task to the patMetModuleTask of the toolCode function
            patMetModuleTask.add(getattr(process, miniAODConfiguration_label))

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
            jetCorLabelUpToL3Name += "Puppi"
            jetCorLabelL3ResName += "Puppi"

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
    def jetCleaning(self, process, jetCollectionName, autoJetCleaning, jetProductionTask, postfix ):

        if autoJetCleaning == "None" or autoJetCleaning == "Manual" :
            return cms.InputTag(jetCollectionName)

        #retrieve collections
        electronCollection = self._parameters["electronCollection"].value
        muonCollection = self._parameters["muonCollection"].value
        photonCollection = self._parameters["photonCollection"].value
        tauCollection = self._parameters["tauCollection"].value

        jetCleaning_task, jetCleaning_label = cms.Task(), "jetCleaning_task{}".format(postfix)

        if autoJetCleaning == "Full" : # auto clean taus, photons and jets
            if isValidInputTag(tauCollection):
                process.load("PhysicsTools.PatAlgos.cleaningLayer1.tauCleaner_cfi")
                jetCleaning_task.add(process.cleanPatTaus)
                cleanPatTauProducer = getattr(process, "cleanPatTaus").clone(
                    src = tauCollection
                    )
                cleanPatTauProducer.checkOverlaps.electrons.src = electronCollection
                cleanPatTauProducer.checkOverlaps.muons.src = muonCollection
                addToProcessAndTask("cleanedPatTaus"+postfix, cleanPatTauProducer, process, jetCleaning_task)
                tauCollection = cms.InputTag("cleanedPatTaus"+postfix)

            if isValidInputTag(photonCollection):
                process.load("PhysicsTools.PatAlgos.cleaningLayer1.photonCleaner_cfi")
                jetCleaning_task.add(process.cleanPatPhotons)
                cleanPatPhotonProducer = getattr(process, "cleanPatPhotons").clone(
                    src = photonCollection
                    )
                cleanPatPhotonProducer.checkOverlaps.electrons.src = electronCollection
                addToProcessAndTask("cleanedPatPhotons"+postfix, cleanPatPhotonProducer, process, jetCleaning_task)
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

        addToProcessAndTask("cleanedPatJets"+postfix, cleanPatJetProducer, process, jetCleaning_task)

        addTaskToProcess(process, jetCleaning_label, jetCleaning_task)
        jetProductionTask.add(getattr(process, jetCleaning_label))
        return cms.InputTag("cleanedPatJets"+postfix)

    # function to implement the 2017 EE noise mitigation fix
    def runFixEE2017(self, process, params, jets, cands, goodcolls, patMetModuleTask, postfix):
        # create a local task and a corresponding label to collect all the modules added in this function
        runFixEE2017_task, runFixEE2017_label = cms.Task(), "runFixEE2017_task{}".format(postfix)

        pfCandidateJetsWithEEnoise = _modbad.BadPFCandidateJetsEEnoiseProducer.clone(
            jetsrc = jets,
            userawPt = params["userawPt"],
            ptThreshold = params["ptThreshold"],
            minEtaThreshold = params["minEtaThreshold"],
            maxEtaThreshold = params["maxEtaThreshold"],
        )
        addToProcessAndTask("pfCandidateJetsWithEEnoise"+postfix, pfCandidateJetsWithEEnoise, process, runFixEE2017_task)

        pfcandidateClustered = cms.EDProducer("CandViewMerger",
            src = cms.VInputTag(goodcolls+[jets])
        )
        addToProcessAndTask("pfcandidateClustered"+postfix, pfcandidateClustered, process, runFixEE2017_task)

        pfcandidateForUnclusteredUnc = _mod.candPtrProjector.clone(
            src  = cands,
            veto = "pfcandidateClustered"+postfix,
        )
        addToProcessAndTask("pfcandidateForUnclusteredUnc"+postfix, pfcandidateForUnclusteredUnc, process, runFixEE2017_task)

        badUnclustered = cms.EDFilter("CandPtrSelector",
            src = cms.InputTag("pfcandidateForUnclusteredUnc"+postfix),
            cut = cms.string("abs(eta) > "+str(params["minEtaThreshold"])+" && abs(eta) < "+str(params["maxEtaThreshold"])),
        )
        addToProcessAndTask("badUnclustered"+postfix, badUnclustered, process, runFixEE2017_task)

        blobUnclustered = cms.EDProducer("UnclusteredBlobProducer",
            candsrc = cms.InputTag("badUnclustered"+postfix),
        )
        addToProcessAndTask("blobUnclustered"+postfix, blobUnclustered, process, runFixEE2017_task)

        superbad = cms.EDProducer("CandViewMerger",
            src = cms.VInputTag(
                cms.InputTag("blobUnclustered"+postfix),
                cms.InputTag("pfCandidateJetsWithEEnoise"+postfix,"bad"),
            )
        )
        addToProcessAndTask("superbad"+postfix, superbad, process, runFixEE2017_task)

        pfCandidatesGoodEE2017 = _mod.candPtrProjector.clone(
            src  = cands,
            veto = "superbad"+postfix,
        )
        addToProcessAndTask("pfCandidatesGoodEE2017"+postfix, pfCandidatesGoodEE2017, process, runFixEE2017_task)

        # add the local task to the process
        addTaskToProcess(process, runFixEE2017_label, runFixEE2017_task)

        # add the task to the patMetModuleTask of the toolCode function
        patMetModuleTask.add(getattr(process, runFixEE2017_label))

        # return good cands and jets
        return (cms.InputTag("pfCandidatesGoodEE2017"+postfix), cms.InputTag("pfCandidateJetsWithEEnoise"+postfix,"good"))

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
    runMETCorrectionsAndUncertainties(process, metType=metType,
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
    runMETCorrectionsAndUncertainties(process, metType=metType,
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
    runMETCorrectionsAndUncertainties(process, metType=metType,
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
                               reclusterJets=None,
                               jetSelection="pt>15 && abs(eta)<9.9",
                               recoMetFromPFCs=None,
                               jetCorLabelL3="ak4PFCHSL1FastL2L3Corrector",
                               jetCorLabelRes="ak4PFCHSL1FastL2L3ResidualCorrector",
##                               jecUncFile="CondFormats/JetMETObjects/data/Summer15_50nsV5_DATA_UncertaintySources_AK4PFchs.txt",
                               CHS=False,
                               puppiProducerLabel="puppi",
                               puppiProducerForMETLabel="puppiNoLep",
                               reapplyJEC=True,
                               jecUncFile="",
                               computeMETSignificance=True,
                               fixEE2017=False,
                               fixEE2017Params=None,
                               extractDeepMETs=False,
                               campaign="",
                               era="",
                               postfix=""):

    runMETCorrectionsAndUncertainties = RunMETCorrectionsAndUncertainties()

    #MET T1 uncertainties
    runMETCorrectionsAndUncertainties(process, metType=metType,
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
                                      reapplyJEC=reapplyJEC,
                                      reclusterJets=reclusterJets,
                                      jetSelection=jetSelection,
                                      recoMetFromPFCs=recoMetFromPFCs,
                                      autoJetCleaning=jetCleaning,
                                      manualJetConfig=manualJetConfig,
                                      jetFlavor=jetFlavor,
                                      jetCorLabelUpToL3=jetCorLabelL3,
                                      jetCorLabelL3Res=jetCorLabelRes,
                                      jecUncertaintyFile=jecUncFile,
                                      computeMETSignificance=computeMETSignificance,
                                      CHS=CHS,
                                      puppiProducerLabel=puppiProducerLabel,
                                      puppiProducerForMETLabel=puppiProducerForMETLabel,
                                      postfix=postfix,
                                      fixEE2017=fixEE2017,
                                      fixEE2017Params=fixEE2017Params,
                                      extractDeepMETs=extractDeepMETs,
                                      campaign=campaign,
                                      era=era,
                                      )

    #MET T1+Txy / Smear
    runMETCorrectionsAndUncertainties(process, metType=metType,
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
                                      reapplyJEC=reapplyJEC,
                                      reclusterJets=reclusterJets,
                                      jetSelection=jetSelection,
                                      recoMetFromPFCs=recoMetFromPFCs,
                                      autoJetCleaning=jetCleaning,
                                      manualJetConfig=manualJetConfig,
                                      jetFlavor=jetFlavor,
                                      jetCorLabelUpToL3=jetCorLabelL3,
                                      jetCorLabelL3Res=jetCorLabelRes,
                                      jecUncertaintyFile=jecUncFile,
                                      computeMETSignificance=computeMETSignificance,
                                      CHS=CHS,
                                      puppiProducerLabel=puppiProducerLabel,
                                      puppiProducerForMETLabel=puppiProducerForMETLabel,
                                      postfix=postfix,
                                      fixEE2017=fixEE2017,
                                      fixEE2017Params=fixEE2017Params,
                                      extractDeepMETs=extractDeepMETs,
                                      campaign=campaign,
                                      era=era,
                                      )
    #MET T1+Smear + uncertainties
    runMETCorrectionsAndUncertainties(process, metType=metType,
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
                                      reapplyJEC=reapplyJEC,
                                      reclusterJets=reclusterJets,
                                      jetSelection=jetSelection,
                                      recoMetFromPFCs=recoMetFromPFCs,
                                      autoJetCleaning=jetCleaning,
                                      manualJetConfig=manualJetConfig,
                                      jetFlavor=jetFlavor,
                                      jetCorLabelUpToL3=jetCorLabelL3,
                                      jetCorLabelL3Res=jetCorLabelRes,
                                      jecUncertaintyFile=jecUncFile,
                                      computeMETSignificance=computeMETSignificance,
                                      CHS=CHS,
                                      puppiProducerLabel=puppiProducerLabel,
                                      puppiProducerForMETLabel=puppiProducerForMETLabel,
                                      postfix=postfix,
                                      fixEE2017=fixEE2017,
                                      fixEE2017Params=fixEE2017Params,
                                      extractDeepMETs=extractDeepMETs,
                                      campaign=campaign,
                                      era=era,
                                      )
