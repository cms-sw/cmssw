import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.ConfigToolBase import *

import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.trigTools import _addEventContent
from PhysicsTools.PatUtils.tools.jmeUncertaintyTools import JetMEtUncertaintyTools

from PhysicsTools.PatUtils.patPFMETCorrections_cff import *
import RecoMET.METProducers.METSigParams_cfi as jetResolutions
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs

class RunJetUncertainties(JetMEtUncertaintyTools):

    """ Produce collection of pat::Jets with jet energy and resolution shifted up/down,
        in order to estimate effect of systematic uncertainties
    """
    _label='runJetUncertainties'
    _defaultParameters = dicttypes.SortedKeysDict()
    def __init__(self):
        JetMEtUncertaintyTools.__init__(self)
        self.addParameter(self._defaultParameters, 'dRjetCleaning', 0.5, 
                          "Eta-phi distance for extra jet cleaning", Type=float)
        self._parameters = copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def __call__(self, process,
                 electronCollection      = None,
                 photonCollection        = None,
                 muonCollection          = None,
                 tauCollection           = None,
                 jetCollection           = None,
                 dRjetCleaning           = None,
                 jetCorrLabel            = None,
                 doSmearJets             = None,
                 jetSmearFileName        = None,
                 jetSmearHistogram       = None,
                 jetCorrPayloadName      = None,
                 jetCorrLabelUpToL3      = None,
                 jetCorrLabelUpToL3Res   = None,
                 jecUncertaintyFile      = None,
                 jecUncertaintyTag       = None,
                 varyByNsigmas           = None,
                 addToPatDefaultSequence = None,
                 outputModule            = None,
                 postfix                 = None):
        JetMEtUncertaintyTools.__call__(
            self, process,
            electronCollection = electronCollection,
            photonCollection = photonCollection,
            muonCollection = muonCollection,
            tauCollection = tauCollection,
            jetCollection = jetCollection,
            jetCorrLabel = jetCorrLabel,
            doSmearJets = doSmearJets,
            jetSmearFileName = jetSmearFileName,
            jetSmearHistogram = jetSmearHistogram,
            jetCorrPayloadName = jetCorrPayloadName,
            jetCorrLabelUpToL3 = jetCorrLabelUpToL3,
            jetCorrLabelUpToL3Res = jetCorrLabelUpToL3Res,
            jecUncertaintyFile = jecUncertaintyFile,
            jecUncertaintyTag = jecUncertaintyTag,
            varyByNsigmas = varyByNsigmas,
            addToPatDefaultSequence = addToPatDefaultSequence,
            outputModule = outputModule,
            postfix = postfix)
        if dRjetCleaning is None:
            dRjetCleaning = self._defaultParameters['dRjetCleaning'].value

        self.setParameter('dRjetCleaning', dRjetCleaning)

        self.apply(process) 

    def toolCode(self, process):        
        electronCollection = self._parameters['electronCollection'].value
        photonCollection = self._parameters['photonCollection'].value
        muonCollection = self._parameters['muonCollection'].value
        tauCollection = self._parameters['tauCollection'].value
        jetCollection = self._parameters['jetCollection'].value
        jetCorrLabel = self._parameters['jetCorrLabel'].value
        dRjetCleaning =  self._parameters['dRjetCleaning'].value
        doSmearJets = self._parameters['doSmearJets'].value
        jetSmearFileName = self._parameters['jetSmearFileName'].value
        jetSmearHistogram = self._parameters['jetSmearHistogram'].value
        jetCorrPayloadName = self._parameters['jetCorrPayloadName'].value
        jetCorrLabelUpToL3 = self._parameters['jetCorrLabelUpToL3'].value
        jetCorrLabelUpToL3Res = self._parameters['jetCorrLabelUpToL3Res'].value
        jecUncertaintyFile = self._parameters['jecUncertaintyFile'].value
        jecUncertaintyTag = self._parameters['jecUncertaintyTag'].value
        varyByNsigmas = self._parameters['varyByNsigmas'].value
        addToPatDefaultSequence = self._parameters['addToPatDefaultSequence'].value
        outputModule = self._parameters['outputModule'].value
        postfix = self._parameters['postfix'].value

        if not hasattr(process, "jetUncertaintySequence" + postfix):
            jetUncertaintySequence = cms.Sequence()
            setattr(process, "jetUncertaintySequence" + postfix, jetUncertaintySequence)
        jetUncertaintySequence = getattr(process, "jetUncertaintySequence" + postfix)

        collectionsToKeep = []

        # produce collection of jets not overlapping with reconstructed
        # electrons/photons, muons and tau-jet candidates
        lastJetCollection, cleanedJetCollection = \
            self._addCleanedJets(process, jetCollection,
                                 electronCollection, photonCollection, muonCollection, tauCollection,
                                 jetUncertaintySequence, postfix)

        # smear jet energies to account for difference in jet resolutions between MC and Data
        # (cf. JME-10-014 PAS)        
        jetCollectionResUp = None
        jetCollectionResDown = None
        if doSmearJets:
            lastJetCollection = \
              self._addSmearedJets(process, cleanedJetCollection, [ "smeared", jetCollection.value() ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas,
                                   uncertaintySequence = jetUncertaintySequence, postfix = postfix)                
            jetCollectionResUp = \
              self._addSmearedJets(process, cleanedJetCollection, [ "smeared", jetCollection.value(), "ResUp" ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas, -1., 
                                   uncertaintySequence = jetUncertaintySequence, postfix = postfix)
            collectionsToKeep.append(jetCollectionResUp)
            jetCollectionResDown = \
              self._addSmearedJets(process, cleanedJetCollection, [ "smeared", jetCollection.value(), "ResDown" ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas, +1., 
                                   uncertaintySequence = jetUncertaintySequence, postfix = postfix)
            collectionsToKeep.append(jetCollectionResDown)

        collectionsToKeep.append(lastJetCollection)

        #--------------------------------------------------------------------------------------------    
        # produce collection of electrons/photons, muons, tau-jet candidates and jets
        # shifted up/down in energy by their respective energy uncertainties
        #--------------------------------------------------------------------------------------------

        shiftedParticleSequence, shiftedParticleCollections, addCollectionsToKeep = \
          self._addShiftedParticleCollections(process,
                                              None,
                                              None,
                                              None,
                                              None,
                                              jetCollection, cleanedJetCollection, lastJetCollection,
                                              jetCollectionResUp, jetCollectionResDown,
                                              jetCorrLabelUpToL3, jetCorrLabelUpToL3Res,
                                              jecUncertaintyFile, jecUncertaintyTag,
                                              varyByNsigmas,
                                              postfix)
        setattr(process, "shiftedParticlesForJetUncertainties" + postfix, shiftedParticleSequence)        
        jetUncertaintySequence += getattr(process, "shiftedParticlesForJetUncertainties" + postfix)
        collectionsToKeep.extend(addCollectionsToKeep)

        # insert metUncertaintySequence into patDefaultSequence
        if addToPatDefaultSequence:
            if not hasattr(process, "patDefaultSequence"):
                raise ValueError("PAT default sequence is not defined !!")
            process.patDefaultSequence += jetUncertaintySequence        

        # add shifted + unshifted collections pf pat::Electrons/Photons,
        # Muons, Taus, Jets and MET to PAT-tuple event content
        if outputModule is not None and hasattr(process, outputModule):
            getattr(process, outputModule).outputCommands = _addEventContent(
                getattr(process, outputModule).outputCommands,
                [ 'keep *_%s_*_%s' % (collectionToKeep, process.name_()) for collectionToKeep in collectionsToKeep ])

runJetUncertainties = RunJetUncertainties()
