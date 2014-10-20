import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *

import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.trigTools import _addEventContent
from PhysicsTools.PatUtils.tools.jmeUncertaintyTools import JetMEtUncertaintyTools
from PhysicsTools.PatUtils.tools.objectsUncertaintyTools import isValidInputTag,addSmearedJets

import RecoMET.METProducers.METSigParams_cfi as jetResolutions



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

    def _addPFMVAMEt(self, process, metUncertaintySequence,
                       shiftedParticleCollections, pfCandCollection,
                       collectionsToKeep,
                       doSmearJets,
                       jecUncertaintyFile, jecUncertaintyTag,
                       varyByNsigmas,
                       postfix):

        # loading default files

        if not hasattr(process, "pfMEtMVA"):
            process.load("RecoMET.METPUSubtraction.mvaPFMET_cff")

        pfCandCollectionUnsmeared = pfCandCollection
        lastUncorrectedJetCollection = 'ak4PFJets'
        lastCorrectedJetCollection = 'calibratedAK4PFJetsForPFMVAMEt'

        #set postfix if not empty value
        if postfix != "":
            configtools.cloneProcessingSnippet(process, process.pfMVAMEtSequence, postfix)
            lastCorrectedJetCollection += postfix

        if doSmearJets:
            self._createUncorrectedJetsModules(process, jetResolutions, pfCandCollection, metUncertaintySequence, postfix)
        else:
            metUncertaintySequence += getattr(process, "pfMVAMEtSequence" + postfix)
        self._addPATMEtProducer(process, metUncertaintySequence,
                                'pfMVAMEt' + postfix, 'patPFMVAMEt', collectionsToKeep, postfix)


        variations=['Up','Down']
        varDir= { 'Up':1., 'Down':-1. }

        #=====================================================
        # Leptons
        #=====================================================
        for leptonCollection in [ [ 'Electron', 'En', 'electronCollection', 0.3 ],
                                  [ 'Photon',   'En', 'photonCollection',   0.3 ],
                                  [ 'Muon',     'En', 'muonCollection',     0.3 ],
                                  [ 'Tau',      'En', 'tauCollection',      0.3 ] ]:



           # pfCandCollectionLeptonShift= { 'Up':None, 'Down':None }

            if ( leptonCollection[2] in shiftedParticleCollections ) and isValidInputTag(shiftedParticleCollections[leptonCollection[2]]):
                pfCandCollectionLeptonShift = \
                  self._addPFCandidatesForPFMEtInput(
                    process, metUncertaintySequence,
                    shiftedParticleCollections['%s' % leptonCollection[2]], leptonCollection[0], leptonCollection[1],
                    shiftedParticleCollections['%s%sUp' % (leptonCollection[2], leptonCollection[1])],
                    shiftedParticleCollections['%s%sDown' % (leptonCollection[2], leptonCollection[1])],
                    leptonCollection[3],
                    pfCandCollection, postfix)


                for var in variations:

                      modulePFMEtLeptonShift = getattr(process, "pfMVAMEt" + postfix).clone(
                        srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShift[var]),
                        srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                        shiftedParticleCollections, leptonCollection[2], '%s%s%s' % (leptonCollection[2], leptonCollection[1], var), postfix = postfix))
                      )
                      modulePFMEtLeptonShiftName = "pfMVAMEt%s%s%s" % (leptonCollection[0], leptonCollection[1],var)
                      modulePFMEtLeptonShiftName += postfix
                      setattr(process, modulePFMEtLeptonShiftName, modulePFMEtLeptonShift)
                      metUncertaintySequence += modulePFMEtLeptonShift
                      self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftName, 'patPFMVAMet%s%s%s' % (leptonCollection[0], leptonCollection[1],var), collectionsToKeep, postfix)

        #=====================================================
        # Jets
        #=====================================================

        # energy shift =======================
        for var in variations:

            if isValidInputTag(shiftedParticleCollections['jetCollection']):
                setattr(process, "uncorrectedJetsEn%sForPFMVAMEt%s" % (var, postfix), cms.EDProducer("ShiftedPFJetProducer",
                           src = cms.InputTag(lastUncorrectedJetCollection),
                           jetCorrInputFileName = cms.FileInPath(jecUncertaintyFile),
                           jetCorrUncertaintyTag = cms.string(jecUncertaintyTag),
                           addResidualJES = cms.bool(False),
                           jetCorrLabelUpToL3 = cms.InputTag("ak4PFL1FastL2L3Corrector"),
                           jetCorrLabelUpToL3Res = cms.InputTag("ak4PFL1FastL2L3ResidualCorrector"),
                           shiftBy = cms.double( varDir[var] *varyByNsigmas)
                          ))
                metUncertaintySequence += getattr(process, "uncorrectedJetsEn%sForPFMVAMEt%s" % (var, postfix) )

                setattr(process, "correctedJetsEn%sForPFMVAMEt%s" %(var, postfix), getattr(process, "uncorrectedJetsEn%sForPFMVAMEt%s" % (var, postfix) ).clone(
                            src = cms.InputTag(lastCorrectedJetCollection),
                          ))
                metUncertaintySequence += getattr(process, "correctedJetsEn%sForPFMVAMEt%s" % (var, postfix) )

  #      pfCandCollectionJetEnShift= { 'Up':None, 'Down':None }
  #      pfCandCollectionJetResShift= { 'Up':None, 'Down':None }
        pfCandCollectionJetEnShift  = \
            self._addPFCandidatesForPFMEtInput(
                process, metUncertaintySequence,
                shiftedParticleCollections['lastJetCollection'], "Jet", "En",
                shiftedParticleCollections['jetCollectionEnUp'], shiftedParticleCollections['jetCollectionEnDown'],
                0.5,
                pfCandCollection, postfix)


        # energy resolution shift =======================
        if hasattr(process, "smearedUncorrectedJetsForPFMVAMEt" + postfix):
       #     pfCandCollectionJetResShift[ 'Up' ], pfCandCollectionJetResShift[ 'Down' ] = \
       #           self._addPFCandidatesForPFMEtInput(
       #             process, metUncertaintySequence,
       #             shiftedParticleCollections['jetCollection'], "Jet", "Res",
       #             shiftedParticleCollections['jetCollectionResUp'], shiftedParticleCollections['jetCollectionResDown'],
       #             0.5,
       #             pfCandCollectionUnsmeared, postfix)

            for var in variations:
                setattr(process, "pfMVAMEtJetEn" + var + postfix, getattr(process, "pfMVAMEt").clone(
                   srcCorrJets = cms.InputTag('correctedJetsEn%sForPFMVAMEt%s' % (var, postfix) ),
                   srcUncorrJets = cms.InputTag('uncorrectedJetsEn%sForPFMVAMEt%s' %(var, postfix) ),
                   srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
                ))
                metUncertaintySequence += getattr(process, "pfMVAMEtJetEn" + var + postfix)
                self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMVAMEtJetEn' + var + postfix, 'patMVAPFMetJetEn' + var, collectionsToKeep, postfix)


                setattr(process, "uncorrectedJetsRes%sForPFMVAMEt%s"% (var, postfix), getattr(process, "smearedUncorrectedJetsForPFMVAMEt" + postfix).clone(
                    shiftBy = cms.double(varDir[var]*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, "uncorrectedJetsRes%sForPFMVAMEt%s" % (var, postfix) )
                setattr(process, "smearedPFCandidatesJetRes%sForPFMVAMEt%s" % (var, postfix), getattr(process, "smearedPFCandidatesForPFMVAMEt" + postfix).clone(
                    srcShiftedObjects   = cms.InputTag("uncorrectedJetsRes%sForPFMVAMEt%s"  % (var, postfix) ),
                ))
                metUncertaintySequence += getattr(process, "smearedPFCandidatesJetRes%sForPFMVAMEt%s" % (var, postfix))

                setattr(process, "correctedJetsRes%sForPFMVAMEt%s" % (var, postfix), getattr(process, "calibratedAK4PFJetsForPFMVAMEt" + postfix).clone(
                    src = cms.InputTag("uncorrectedJetsRes%sForPFMVAMEt%s" % (var, postfix) )
                ))
                metUncertaintySequence += getattr(process, "correctedJetsRes%sForPFMVAMEt%s" % (var, postfix) )


                setattr(process, "pfMVAMEtJetRes" + var + postfix, getattr(process, "pfMVAMEt" + postfix).clone(
                    srcCorrJets = cms.InputTag('correctedJetsRes%sForPFMVAMEt%s' % (var, postfix)),
                    srcUncorrJets = cms.InputTag('uncorrectedJetsRes%sForPFMVAMEt%s' % (var, postfix)),
                    ##srcPFCandidates = cms.InputTag(pfCandCollectionJetResShift[ var ]),
                    srcPFCandidates = cms.InputTag("smearedPFCandidatesJetRes%sForPFMVAMEt%s" %(var, postfix) ),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
                ))
                metUncertaintySequence += getattr(process, "pfMVAMEtJetRes" + var + postfix)
                self._addPATMEtProducer(process, metUncertaintySequence,
                                       'pfMVAMEtJetRes' + var + postfix, 'patPFMVAMetJetRes' + var, collectionsToKeep, postfix)


        #=====================================================
        #unclustered energy
        #=====================================================

        # check existence of pfCandsNotInJet collection => no need of postfix, they are the same everywhere
        if not hasattr(process, "pfCandsNotInJetsForMetCorr"):
            from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import pfJetsPtrForMetCorr, pfCandsNotInJetsPtrForMetCorr, pfCandsNotInJetsForMetCorr
            metUncertaintySequence += pfJetsPtrForMetCorr
            metUncertaintySequence += pfCandsNotInJetsPtrForMetCorr
            metUncertaintySequence += pfCandsNotInJetsForMetCorr


      #  pfCandCollectionUnclusteredEnShift= { 'Up':None, 'Down':None }
        for var in variations:
            setattr(process, "pfCandsNotInJetsUnclusteredEn%sForPFMVAMEt%s" % (var, postfix), cms.EDProducer("ShiftedPFCandidateProducer",
                src = cms.InputTag('pfCandsNotInJetsForMetCorr'),
                shiftBy = cms.double(varDir[var]*varyByNsigmas),
                uncertainty = cms.double(0.10)
            ))
            metUncertaintySequence += getattr(process, "pfCandsNotInJetsUnclusteredEn%sForPFMVAMEt%s" % (var, postfix))

        pfCandCollectionUnclusteredEnShift = \
            self._addPFCandidatesForPFMEtInput(
            process, metUncertaintySequence,
            pfCandCollection, "Unclustered", "En",
            'pfCandsNotInJetsUnclusteredEnUpForPFMVAMEt' + postfix, 'pfCandsNotInJetsUnclusteredEnDownForPFMVAMEt' + postfix,
            0.01,
            pfCandCollection, postfix)

        for var in variations:
            setattr(process, "pfMVAMEtUnclusteredEn" + var + postfix, getattr(process, "pfMVAMEt" + postfix).clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionUnclusteredEnShift[var]),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
            ))
            metUncertaintySequence += getattr(process, "pfMVAMEtUnclusteredEn" + var + postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMVAMEtUnclusteredEn' + var + postfix, 'patPFMVAMetUnclusteredEn'+var, collectionsToKeep, postfix)





    def _createUncorrectedJetsModules(self,process, jetResolutions, pfCandCollection, metUncertaintySequence, postfix):

        process.load("RecoJets.Configuration.GenJetParticles_cff")
        metUncertaintySequence += process.genParticlesForJetsNoNu
        process.load("RecoJets.Configuration.RecoGenJets_cff")
        metUncertaintySequence += process.ak4GenJetsNoNu
        setattr(process, "smearedUncorrectedJetsForPFMVAMEt" + postfix, cms.EDProducer("SmearedPFJetProducer",
                                                   src = cms.InputTag('ak4PFJets'),
                                                   jetCorrLabel = cms.InputTag("ak4PFL1FastL2L3Corrector"),
                                                   dRmaxGenJetMatch = cms.string('min(0.5, 0.1 + 0.3*exp(-0.05*(genJetPt - 10.)))'),
                                                   sigmaMaxGenJetMatch = cms.double(3.),
                                                   inputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/pfJetResolutionMCtoDataCorrLUT.root'),
                                                   lutName = cms.string('pfJetResolutionMCtoDataCorrLUT'),
                                                   jetResolutions = jetResolutions.METSignificance_params,
                                                   skipRawJetPtThreshold = cms.double(10.), # GeV
                                                   skipCorrJetPtThreshold = cms.double(1.e-2),
                                                   srcGenJets = cms.InputTag('ak4GenJetsNoNu'),
                                                   #verbosity = cms.int32(1)
                                                   ))
        setattr(process, "smearedPFCandidatesForPFMVAMEt" + postfix, cms.EDProducer("ShiftedPFCandidateProducerForPFMVAMEt",
                                                   srcPFCandidates     = pfCandCollection,
                                                   srcUnshiftedObjects = cms.InputTag("ak4PFJets"),
                                                   srcShiftedObjects   = cms.InputTag("smearedUncorrectedJetsForPFMVAMEt" + postfix),
                                                   dRmatch_PFCandidate = cms.double(0.5)
                                                   ))
        pfCandCollection = cms.InputTag("smearedPFCandidatesForPFMVAMEt" + postfix)
        metUncertaintySequence += getattr(process, "smearedUncorrectedJetsForPFMVAMEt" + postfix)
        metUncertaintySequence += getattr(process, "smearedPFCandidatesForPFMVAMEt" + postfix)
        getattr(process, "calibratedAK4PFJetsForPFMVAMEt" + postfix).src = cms.InputTag('smearedUncorrectedJetsForPFMVAMEt' + postfix)
        getattr(process, "pfMVAMEt" + postfix).srcUncorrJets   = cms.InputTag('smearedUncorrectedJetsForPFMVAMEt' + postfix)
        getattr(process, "pfMVAMEt" + postfix).srcPFCandidates = cms.InputTag('smearedPFCandidatesForPFMVAMEt' + postfix)
        metUncertaintySequence += getattr(process, "calibratedAK4PFJetsForPFMVAMEt" + postfix)
        metUncertaintySequence += getattr(process, "pfMVAMEt" + postfix)
        lastUncorrectedJetCollection = 'smearedUncorrectedJetsForPFMVAMEt' + postfix


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
        if isValidInputTag(jetCollection):
            if doSmearJets:
                lastJetCollection = \
                    addSmearedJets(process, lastJetCollection, [ "smeared", jetCollection.value(), "ForPFMVAMEt" ],
                                   jetSmearFileName, jetSmearHistogram, jetResolutions, varyByNsigmas, None,
                                   sequence = metUncertaintySequence, postfix = postfix)
#            jetCollectionResUp = \
#              self._addSmearedJets(process, lastJetCollection, [ "smeared", jetCollection.value(), "ForPFMVAMEtResUp" ],
#                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas, -1.,
#                                   uncertaintySequence = metUncertaintySequence, postfix = postfix)
#            collectionsToKeep.append(jetCollectionResUp)
#            jetCollectionResDown = \
#              self._addSmearedJets(process, lastJetCollection, [ "smeared", jetCollection.value(), "ForPFMVAMEtResDown" ],
#                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas, +1.,
#                                   uncertaintySequence = metUncertaintySequence, postfix = postfix)
#            collectionsToKeep.append(jetCollectionResDown)

            collectionsToKeep.append(lastJetCollection)

        #--------------------------------------------------------------------------------------------
        # produce collection of electrons/photons, muons, tau-jet candidates and jets
        # shifted up/down in energy by their respective energy uncertainties
        #--------------------------------------------------------------------------------------------
        shiftedParticleSequence, shiftedParticleCollections, addCollectionsToKeep = \
          self._addShiftedParticleCollections(process,
                                              electronCollection.value(),
                                              photonCollection.value(),
                                              muonCollection.value(),
                                              tauCollection.value(),
                                              jetCollection.value(), lastJetCollection, lastJetCollection,
                                              doSmearJets,
                                              jetCorrLabelUpToL3, jetCorrLabelUpToL3Res,
                                              jecUncertaintyFile, jecUncertaintyTag,
                                              jetSmearFileName, jetSmearHistogram,
                                              varyByNsigmas,
                                              "ForPFMVAMEt" + postfix)
        setattr(process, "shiftedParticlesForPFMVAMEtUncertainties" + postfix, shiftedParticleSequence)
        metUncertaintySequence += getattr(process, "shiftedParticlesForPFMVAMEtUncertainties" + postfix)
        collectionsToKeep.extend(addCollectionsToKeep)

        #--------------------------------------------------------------------------------------------
        # propagate shifted particle energies to MVA MET
        #--------------------------------------------------------------------------------------------

        self._addPFMVAMEt(process, metUncertaintySequence,
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
