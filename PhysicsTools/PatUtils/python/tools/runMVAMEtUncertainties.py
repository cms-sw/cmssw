import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *

import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.trigTools import _addEventContent
from PhysicsTools.PatUtils.tools.jmeUncertaintyTools import JetMEtUncertaintyTools

from PhysicsTools.PatUtils.patPFMETCorrections_cff import *
import RecoMET.METProducers.METSigParams_cfi as jetResolutions
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs
 
class RunMVAMEtUncertainties(JetMEtUncertaintyTools):

    """ Shift energy of electrons, photons, muons, tau-jets and other jets
    reconstructed in the event up/down,
    in order to estimate effect of energy scale uncertainties on MVA MET
   """
    _label = 'runMVAMEtUncertainties'
    _defaultParameters = dicttypes.SortedKeysDict()
    def __init__(self):
        JetMEtUncertaintyTools.__init__(self)        
	self.addParameter(self._defaultParameters, 'pfCandCollection', cms.InputTag('particleFlow'), 
                          "Input PFCandidate collection", Type = cms.InputTag)	
        self._parameters = copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def _addPFMEtByMVA(self, process, metUncertaintySequence,
                       shiftedParticleCollections, pfCandCollection,
                       collectionsToKeep,
                       doSmearJets,
                       jecUncertaintyFile, jecUncertaintyTag,
                       varyByNsigmas,
                       postfix):

        if not hasattr(process, "pfMEtMVA"):
            process.load("JetMETCorrections.METPUSubtraction.mvaPFMET_cff")

        lastUncorrectedJetCollection = 'ak5PFJets'
        lastCorrectedJetCollection = 'calibratedAK5PFJetsForPFMEtMVA'
        if postfix != "":
            configtools.cloneProcessingSnippet(process, process.pfMEtMVAsequence, postfix)
            lastCorrectedJetCollection += postfix

        if doSmearJets:
            process.load("RecoJets.Configuration.GenJetParticles_cff")
            metUncertaintySequence += process.genParticlesForJetsNoNu
            process.load("RecoJets.Configuration.RecoGenJets_cff")
            metUncertaintySequence += process.ak5GenJetsNoNu
            setattr(process, "smearedUncorrectedJetsForPFMEtByMVA" + postfix, cms.EDProducer("SmearedPFJetProducer",
                src = cms.InputTag('ak5PFJets'),
                jetCorrLabel = cms.string("ak5PFL1FastL2L3"),                                       
                dRmaxGenJetMatch = cms.string('TMath::Min(0.5, 0.1 + 0.3*TMath::Exp(-0.05*(genJetPt - 10.)))'),
                sigmaMaxGenJetMatch = cms.double(5.),                                                               
                inputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/pfJetResolutionMCtoDataCorrLUT.root'),
                lutName = cms.string('pfJetResolutionMCtoDataCorrLUT'),
                jetResolutions = jetResolutions.METSignificance_params,
                skipRawJetPtThreshold = cms.double(10.), # GeV
                skipCorrJetPtThreshold = cms.double(1.e-2),
                srcGenJets = cms.InputTag('ak5GenJetsNoNu')
            ))
            metUncertaintySequence += getattr(process, "smearedUncorrectedJetsForPFMEtByMVA" + postfix)
            getattr(process, "calibratedAK5PFJetsForPFMEtMVA" + postfix).src = cms.InputTag('smearedUncorrectedJetsForPFMEtByMVA' + postfix)
            getattr(process, "pfMEtMVA" + postfix).srcUncorrJets = cms.InputTag('smearedUncorrectedJetsForPFMEtByMVA' + postfix)
            metUncertaintySequence += getattr(process, "calibratedAK5PFJetsForPFMEtMVA" + postfix)
            setattr(process, "smearedCorrectedJetsForPFMEtByMVA" + postfix, getattr(process, "smearedUncorrectedJetsForPFMEtByMVA" + postfix).clone(
                src = cms.InputTag('calibratedAK5PFJetsForPFMEtMVA' + postfix),
                jetCorrLabel = cms.string("")
            ))
            metUncertaintySequence += getattr(process, "smearedCorrectedJetsForPFMEtByMVA" + postfix)
            getattr(process, "pfMEtMVA" + postfix).srcCorrJets = cms.InputTag('smearedCorrectedJetsForPFMEtByMVA' + postfix)
            metUncertaintySequence += getattr(process, "pfMEtMVA" + postfix)
        else:
            metUncertaintySequence += getattr(process, "pfMEtMVAsequence" + postfix)
        self._addPATMEtProducer(process, metUncertaintySequence,
                                'pfMEtMVA' + postfix, 'patPFMetMVA', collectionsToKeep, postfix)

        for leptonCollection in [ [ 'Electron', 'En', 'electronCollection', 0.3 ],
                                  [ 'Photon',   'En', 'photonCollection',   0.3 ],
                                  [ 'Muon',     'En', 'muonCollection',     0.3 ],
                                  [ 'Tau',      'En', 'tauCollection',      0.3 ] ]:
            if self._isValidInputTag(shiftedParticleCollections[leptonCollection[2]]):
                pfCandCollectionLeptonShiftUp, pfCandCollectionLeptonShiftDown = \
                  self._addPFCandidatesForPFMEtInput(
                    process, metUncertaintySequence,
                    shiftedParticleCollections['%s' % leptonCollection[2]], leptonCollection[0], leptonCollection[1],
                    shiftedParticleCollections['%s%sUp' % (leptonCollection[2], leptonCollection[1])],
                    shiftedParticleCollections['%s%sDown' % (leptonCollection[2], leptonCollection[1])],
                    leptonCollection[3],
                    pfCandCollection, postfix)
                modulePFMEtLeptonShiftUp = getattr(process, "pfMEtMVA" + postfix).clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShiftUp),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                      shiftedParticleCollections, leptonCollection[2], '%s%sUp' % (leptonCollection[2], leptonCollection[1]), postfix = postfix))
                )
                modulePFMEtLeptonShiftUpName = "pfMEtMVA%s%sUp" % (leptonCollection[0], leptonCollection[1])
                modulePFMEtLeptonShiftUpName += postfix
                setattr(process, modulePFMEtLeptonShiftUpName, modulePFMEtLeptonShiftUp)
                metUncertaintySequence += modulePFMEtLeptonShiftUp
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftUpName, 'patPFMetMVA%s%sUp' % (leptonCollection[0], leptonCollection[1]), collectionsToKeep, postfix)
                modulePFMEtLeptonShiftDown = getattr(process, "pfMEtMVA" + postfix).clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShiftDown),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                      shiftedParticleCollections, leptonCollection[2], '%s%sDown' % (leptonCollection[2], leptonCollection[1]), postfix = postfix))
                )
                modulePFMEtLeptonShiftDownName = "pfMEtMVA%s%sDown" % (leptonCollection[0], leptonCollection[1])
                modulePFMEtLeptonShiftDownName += postfix
                setattr(process, modulePFMEtLeptonShiftDownName, modulePFMEtLeptonShiftDown)
                metUncertaintySequence += modulePFMEtLeptonShiftDown
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftDownName, 'patPFMetMVA%s%sDown' % (leptonCollection[0], leptonCollection[1]), collectionsToKeep, postfix)

        if self._isValidInputTag(shiftedParticleCollections['jetCollection']):            
            setattr(process, "uncorrectedJetsEnUpForPFMEtByMVA" + postfix, cms.EDProducer("ShiftedPFJetProducer",
                src = cms.InputTag(lastUncorrectedJetCollection),
                jetCorrInputFileName = cms.FileInPath(jecUncertaintyFile),
                jetCorrUncertaintyTag = cms.string(jecUncertaintyTag),
                addResidualJES = cms.bool(True),
                jetCorrLabelUpToL3 = cms.string("ak5PFL1FastL2L3"),
                jetCorrLabelUpToL3Res = cms.string("ak5PFL1FastL2L3Residual"),                               
                shiftBy = cms.double(+1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "uncorrectedJetsEnUpForPFMEtByMVA" + postfix)
            setattr(process, "uncorrectedJetsEnDownForPFMEtByMVA" + postfix, getattr(process, "uncorrectedJetsEnUpForPFMEtByMVA" + postfix).clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "uncorrectedJetsEnDownForPFMEtByMVA" + postfix)
            setattr(process, "correctedJetsEnUpForPFMEtByMVA" + postfix, getattr(process, "uncorrectedJetsEnUpForPFMEtByMVA" + postfix).clone(
                src = cms.InputTag(lastCorrectedJetCollection),
                addResidualJES = cms.bool(False),
                shiftBy = cms.double(+1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "correctedJetsEnUpForPFMEtByMVA" + postfix)
            setattr(process, "correctedJetsEnDownForPFMEtByMVA" + postfix, getattr(process, "correctedJetsEnUpForPFMEtByMVA" + postfix).clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "correctedJetsEnDownForPFMEtByMVA" + postfix)
            pfCandCollectionJetEnUp, pfCandCollectionJetEnDown = \
              self._addPFCandidatesForPFMEtInput(
                process, metUncertaintySequence, 
                shiftedParticleCollections['lastJetCollection'], "Jet", "En",
                shiftedParticleCollections['jetCollectionEnUpForCorrMEt'], shiftedParticleCollections['jetCollectionEnDownForCorrMEt'],
                0.5,
                pfCandCollection, postfix)
            setattr(process, "pfMEtMVAJetEnUp" + postfix, getattr(process, "pfMEtMVA").clone(
                srcCorrJets = cms.InputTag('correctedJetsEnUpForPFMEtByMVA' + postfix),
                srcUncorrJets = cms.InputTag('uncorrectedJetsEnUpForPFMEtByMVA' + postfix),
                srcPFCandidates = cms.InputTag(pfCandCollectionJetEnUp),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
            ))
            metUncertaintySequence += getattr(process, "pfMEtMVAJetEnUp" + postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMEtMVAJetEnUp' + postfix, 'patPFMetMVAJetEnUp', collectionsToKeep, postfix)
            setattr(process, "pfMEtMVAJetEnDown" + postfix, getattr(process, "pfMEtMVA" + postfix).clone(
                srcCorrJets = cms.InputTag('correctedJetsEnDownForPFMEtByMVA' + postfix),
                srcUncorrJets = cms.InputTag('uncorrectedJetsEnDownForPFMEtByMVA' + postfix),
                srcPFCandidates = cms.InputTag(pfCandCollectionJetEnDown),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
            ))
            metUncertaintySequence += getattr(process, "pfMEtMVAJetEnDown" + postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMEtMVAJetEnDown' + postfix, 'patPFMetMVAJetEnDown', collectionsToKeep, postfix)

            if hasattr(process, "smearedUncorrectedJetsForPFMEtByMVA" + postfix):
                setattr(process, "uncorrectedJetsResUpForPFMEtByMVA" + postfix, getattr(process, "smearedUncorrectedJetsForPFMEtByMVA" + postfix).clone(
                    shiftBy = cms.double(-1.*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, "uncorrectedJetsResUpForPFMEtByMVA" + postfix)
                setattr(process, "uncorrectedJetsResDownForPFMEtByMVA" + postfix, getattr(process, "smearedUncorrectedJetsForPFMEtByMVA" + postfix).clone(
                    shiftBy = cms.double(+1.*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, "uncorrectedJetsResDownForPFMEtByMVA" + postfix)
                setattr(process, "correctedJetsResUpForPFMEtByMVA" + postfix, getattr(process, "smearedCorrectedJetsForPFMEtByMVA" + postfix).clone(
                    shiftBy = cms.double(-1.*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, "correctedJetsResUpForPFMEtByMVA" + postfix)
                setattr(process, "correctedJetsResDownForPFMEtByMVA" + postfix, getattr(process, "smearedCorrectedJetsForPFMEtByMVA" + postfix).clone(
                    shiftBy = cms.double(+1.*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, "correctedJetsResDownForPFMEtByMVA" + postfix)
                pfCandCollectionJetResUp, pfCandCollectionJetResDown = \
                  self._addPFCandidatesForPFMEtInput(
                    process, metUncertaintySequence,
                    shiftedParticleCollections['jetCollection'], "Jet", "Res",
                    shiftedParticleCollections['jetCollectionResUp'], shiftedParticleCollections['jetCollectionResDown'],
                    0.5,
                    pfCandCollection, postfix)
                setattr(process, "pfMEtMVAJetResUp" + postfix, getattr(process, "pfMEtMVA" + postfix).clone(
                    srcCorrJets = cms.InputTag('correctedJetsResUpForPFMEtByMVA' + postfix),
                    srcUncorrJets = cms.InputTag('uncorrectedJetsResUpForPFMEtByMVA' + postfix),
                    srcPFCandidates = cms.InputTag(pfCandCollectionJetResUp),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
                ))
                metUncertaintySequence += getattr(process, "pfMEtMVAJetResUp" + postfix)
                self._addPATMEtProducer(process, metUncertaintySequence,
                                       'pfMEtMVAJetResUp' + postfix, 'patPFMetMVAJetResUp', collectionsToKeep, postfix)
                setattr(process, "pfMEtMVAJetResDown" + postfix, getattr(process, "pfMEtMVA" + postfix).clone(
                    srcCorrJets = cms.InputTag('correctedJetsResDownForPFMEtByMVA' + postfix),
                    srcUncorrJets = cms.InputTag('uncorrectedJetsResDownForPFMEtByMVA' + postfix),
                    srcPFCandidates = cms.InputTag(pfCandCollectionJetResDown),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
                ))
                metUncertaintySequence += getattr(process, "pfMEtMVAJetResDown" + postfix)
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        'pfMEtMVAJetResDown' + postfix, 'patPFMetMVAJetResDown', collectionsToKeep, postfix)
                        
            setattr(process, "pfCandsNotInJetUnclusteredEnUpForPFMEtByMVA" + postfix, cms.EDProducer("ShiftedPFCandidateProducer",
                src = cms.InputTag('pfCandsNotInJet'),
                shiftBy = cms.double(+1.*varyByNsigmas),
                uncertainty = cms.double(0.10)
            ))
            metUncertaintySequence += getattr(process, "pfCandsNotInJetUnclusteredEnUpForPFMEtByMVA" + postfix)
            setattr(process, "pfCandsNotInJetUnclusteredEnDownForPFMEtByMVA" + postfix, getattr(process, "pfCandsNotInJetUnclusteredEnUpForPFMEtByMVA" + postfix).clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "pfCandsNotInJetUnclusteredEnDownForPFMEtByMVA" + postfix)
            pfCandCollectionUnclusteredEnUp, pfCandCollectionUnclusteredEnDown = \
              self._addPFCandidatesForPFMEtInput(
                process, metUncertaintySequence,
                pfCandCollection, "Unclustered", "En",
                'pfCandsNotInJetUnclusteredEnUpForPFMEtByMVA' + postfix, 'pfCandsNotInJetUnclusteredEnDownForPFMEtByMVA' + postfix, #fixme MM
                0.01,
                pfCandCollection, postfix)
            setattr(process, "pfMEtMVAUnclusteredEnUp" + postfix, getattr(process, "pfMEtMVA" + postfix).clone(
                srcPFCandidates = cms.InputTag(pfCandCollectionUnclusteredEnUp),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
            ))
            metUncertaintySequence += getattr(process, "pfMEtMVAUnclusteredEnUp" + postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMEtMVAUnclusteredEnUp' + postfix, 'patPFMetMVAUnclusteredEnUp', collectionsToKeep, postfix)
            setattr(process, "pfMEtMVAUnclusteredEnDown" + postfix, getattr(process, "pfMEtMVA" + postfix).clone(
                srcPFCandidates = cms.InputTag(pfCandCollectionUnclusteredEnDown),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
            ))
            metUncertaintySequence += getattr(process, "pfMEtMVAUnclusteredEnDown" + postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMEtMVAUnclusteredEnDown' + postfix, 'patPFMetMVAUnclusteredEnDown', collectionsToKeep, postfix)

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
                 pfCandCollection        = None,
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
        pfCandCollection = self._initializeInputTag(pfCandCollection, 'pfCandCollection')

        self.setParameter('pfCandCollection', pfCandCollection)
  
        self.apply(process) 
        
    def toolCode(self, process):        
        electronCollection = self._parameters['electronCollection'].value
        photonCollection = self._parameters['photonCollection'].value
        muonCollection = self._parameters['muonCollection'].value
        tauCollection = self._parameters['tauCollection'].value
        jetCollection = self._parameters['jetCollection'].value
        jetCorrLabel = self._parameters['jetCorrLabel'].value
        doSmearJets = self._parameters['doSmearJets'].value
        jetSmearFileName = self._parameters['jetSmearFileName'].value
        jetSmearHistogram = self._parameters['jetSmearHistogram'].value
        pfCandCollection = self._parameters['pfCandCollection'].value
        jetCorrPayloadName = self._parameters['jetCorrPayloadName'].value
        jetCorrLabelUpToL3 = self._parameters['jetCorrLabelUpToL3'].value
        jetCorrLabelUpToL3Res = self._parameters['jetCorrLabelUpToL3Res'].value
        jecUncertaintyFile = self._parameters['jecUncertaintyFile'].value
        jecUncertaintyTag = self._parameters['jecUncertaintyTag'].value
        varyByNsigmas = self._parameters['varyByNsigmas'].value
        addToPatDefaultSequence = self._parameters['addToPatDefaultSequence'].value
        outputModule = self._parameters['outputModule'].value
        postfix = self._parameters['postfix'].value

        if not hasattr(process, "pfMVAMEtUncertaintySequence" + postfix):
            metUncertaintySequence = cms.Sequence()
            setattr(process, "pfMVAMEtUncertaintySequence" + postfix, metUncertaintySequence)
        metUncertaintySequence = getattr(process, "pfMVAMEtUncertaintySequence" + postfix)

        collectionsToKeep = []

        lastJetCollection = jetCollection.value()
                
        # smear jet energies to account for difference in jet resolutions between MC and Data
        # (cf. JME-10-014 PAS)        
        jetCollectionResUp = None
        jetCollectionResDown = None
        if doSmearJets:
            lastJetCollection = \
              self._addSmearedJets(process, lastJetCollection, [ "smeared", jetCollection.value(), "ForPFMEtByMVA" ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas,
                                   uncertaintySequence = metUncertaintySequence, postfix = postfix)
            jetCollectionResUp = \
              self._addSmearedJets(process, lastJetCollection, [ "smeared", jetCollection.value(), "ForPFMEtByMVAResUp" ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas, -1., 
                                   uncertaintySequence = metUncertaintySequence, postfix = postfix)
            collectionsToKeep.append(jetCollectionResUp)
            jetCollectionResDown = \
              self._addSmearedJets(process, lastJetCollection, [ "smeared", jetCollection.value(), "ForPFMEtByMVAResDown" ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas, +1., 
                                   uncertaintySequence = metUncertaintySequence, postfix = postfix)
            collectionsToKeep.append(jetCollectionResDown)

        collectionsToKeep.append(lastJetCollection)

        #--------------------------------------------------------------------------------------------    
        # produce collection of electrons/photons, muons, tau-jet candidates and jets
        # shifted up/down in energy by their respective energy uncertainties
        #--------------------------------------------------------------------------------------------

        shiftedParticleSequence, shiftedParticleCollections, addCollectionsToKeep = \
          self._addShiftedParticleCollections(process,
                                              electronCollection,
                                              photonCollection,
                                              muonCollection,
                                              tauCollection,
                                              jetCollection, None, lastJetCollection,
                                              jetCollectionResUp, jetCollectionResDown,
                                              jetCorrLabelUpToL3, jetCorrLabelUpToL3Res,
                                              jecUncertaintyFile, jecUncertaintyTag,
                                              varyByNsigmas,
                                              "ForPFMEtByMVA" + postfix)
        setattr(process, "shiftedParticlesForPFMEtByMVAUncertainties" + postfix, shiftedParticleSequence)    
        metUncertaintySequence += getattr(process, "shiftedParticlesForPFMEtByMVAUncertainties" + postfix)
        collectionsToKeep.extend(addCollectionsToKeep)
        
        #--------------------------------------------------------------------------------------------    
        # propagate shifted particle energies to MVA MET
        #--------------------------------------------------------------------------------------------

        self._addPFMEtByMVA(process, metUncertaintySequence,
                            shiftedParticleCollections, pfCandCollection,
                            collectionsToKeep,
                            doSmearJets,
                            jecUncertaintyFile, jecUncertaintyTag, 
                            varyByNsigmas,
                            postfix)
        
        # insert metUncertaintySequence into patDefaultSequence
        if addToPatDefaultSequence:
            if not hasattr(process, "patDefaultSequence"):
                raise ValueError("PAT default sequence is not defined !!")
            process.patDefaultSequence += metUncertaintySequence        
       
        # add shifted + unshifted collections pf pat::Electrons/Photons,
        # Muons, Taus, Jets and MET to PAT-tuple event content
        if outputModule is not None and hasattr(process, outputModule):
            getattr(process, outputModule).outputCommands = _addEventContent(
                getattr(process, outputModule).outputCommands,
                [ 'keep *_%s_*_%s' % (collectionToKeep, process.name_()) for collectionToKeep in collectionsToKeep ])
       
runMVAMEtUncertainties = RunMVAMEtUncertainties()
