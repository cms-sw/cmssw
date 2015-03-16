import FWCore.ParameterSet.Config as cms

def isValidInputTag(input):
    input_str = input
    if isinstance(input, cms.InputTag):
        input_str = input.value()
    if input is None or input_str == '""':
        return False
    else:
        return True


def addModuleToSequence(process, module, moduleName_parts, sequence, postfix):

    if not len(moduleName_parts) > 0:
        raise ValueError("Empty list !!")

    moduleName = ""

    lastPart = None
    for part in moduleName_parts:
        if part is None or part == "":
            continue

        part = part.replace("selected", "")
        part = part.replace("clean",    "")

        if lastPart is None:
            moduleName += part[0].lower() + part[1:]
            lastPart = part
        else:
            if lastPart[-1].islower() or lastPart[-1].isdigit():
                moduleName += part[0].capitalize() + part[1:]
            else:
                moduleName += part[0].lower() + part[1:]
                lastPart = part

    moduleName += postfix
    setattr(process, moduleName, module)

    sequence += module

    return moduleName



def addShiftedSingleParticleCollection(process, identifier,objectCollection,
                                 varyByNsigmas,sequence, postfix = ""):

    shiftedParticleCollections = {}
    collectionsToKeep = []

    shiftedCollectionName = identifier+"Collection"
    shiftedParticleCollections[ shiftedCollectionName ] = objectCollection



    objectCollectionUp = None
    objectCollectionDown = None

    if isValidInputTag(objectCollection):

        ##
        ## Up variation
        ##
        objectCollectionUp = createShiftedSingleParticleUpModule(process,identifier,
                                            objectCollection,
                                            varyByNsigmas, sequence,postfix)
      #  collectionName = objectCollection.value()
      #  collectionName = collectionName.replace("selected", "")
      #  collectionName = collectionName.replace("clean",    "")
        objectModuleUp = getattr(process,  objectCollectionUp)
        shiftedParticleCollections[ shiftedCollectionName+'EnUp'] = objectCollectionUp
        collectionsToKeep.append(objectCollectionUp)

        ##
        ## Down variation
        ##
        objectModuleDown = objectModuleUp.clone(
            shiftBy = cms.double(-1.*varyByNsigmas)
            )
        objectCollectionDown = addModuleToSequence(process, objectModuleDown,
                                 [ "shifted", objectCollection, "EnDown" ],
                                                   sequence, postfix)
        shiftedParticleCollections[ shiftedCollectionName+'EnDown'] = objectCollectionDown
        collectionsToKeep.append(objectCollectionDown)


        return (shiftedParticleCollections, collectionsToKeep)



def createShiftedSingleParticleUpModule(process,identifier, objectCollection,
                                        varyByNsigmas,sequence,postfix=""):

    shiftedCollectionUp = None

    if identifier == "electron":
        shiftedModuleUp = cms.EDProducer("ShiftedPATElectronProducer",
                src = cms.InputTag(objectCollection),
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

    if identifier == "photon":
        shiftedModuleUp = cms.EDProducer("ShiftedPATPhotonProducer",
                src = cms.InputTag(objectCollection),
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

    if identifier == "muon":
        shiftedModuleUp = cms.EDProducer("ShiftedPATMuonProducer",
                src = cms.InputTag(objectCollection),
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

    if identifier == "tau":
        shiftedModuleUp = cms.EDProducer("ShiftedPATTauProducer",
                src = cms.InputTag(objectCollection),
                uncertainty = cms.double(0.03),
                shiftBy = cms.double(+1.*varyByNsigmas)
            )

    shiftedCollectionUp = addModuleToSequence(process, shiftedModuleUp,
                        [ "shifted", objectCollection, "EnUp" ],
                        sequence, postfix)

    return shiftedCollectionUp




def addShiftedJetCollections(process, jetCollection, lastJetCollection,
                             jetCorrLabelUpToL3, jetCorrLabelUpToL3Res,
                             jecUncertaintyFile, jecUncertaintyTag,
                             varyByNsigmas, sequence,
                             postfix = ""):

    shiftedParticleCollections = {}
    collectionsToKeep = []

    shiftedParticleCollections['jetCollection'] = jetCollection
    shiftedParticleCollections['lastJetCollection'] = lastJetCollection

    #
    # Creating two sets for shifted jets corrections, depending of the use needed :
    #

    variations = { "Up":1., "Down":-1.  }

    for var in variations.keys():
        # in case of "raw" (uncorrected) MET,
        # add residual jet energy corrections in quadrature to jet energy uncertainties:
        # cf. https://twiki.cern.ch/twiki/bin/view/CMS/MissingETUncertaintyPrescription
        jetsEnShiftForRawMEt = cms.EDProducer("ShiftedPATJetProducer",
                                         src = cms.InputTag(lastJetCollection),
                                         #jetCorrPayloadName = cms.string(jetCorrPayloadName),
                                         #jetCorrUncertaintyTag = cms.string('Uncertainty'),
                                         jetCorrInputFileName = cms.FileInPath(jecUncertaintyFile),
                                         jetCorrUncertaintyTag = cms.string(jecUncertaintyTag),
                                         addResidualJES = cms.bool(True),
                                         jetCorrLabelUpToL3 = cms.InputTag(jetCorrLabelUpToL3.value()),
                                         jetCorrLabelUpToL3Res = cms.InputTag(jetCorrLabelUpToL3Res.value()),
                                         shiftBy = cms.double(variations[var]*varyByNsigmas)
                                         )

        jetCollectionEnShiftForRawMEt = \
            addModuleToSequence(process, jetsEnShiftForRawMEt,
                                [ "shifted", jetCollection, "En%sForRawMEt"%var ],
                                sequence, postfix)
        shiftedParticleCollections[ 'jetCollectionEn%sForRawMEt'%var ] = jetCollectionEnShiftForRawMEt
        collectionsToKeep.append(jetCollectionEnShiftForRawMEt)

        ##
        ## Now, jet variation for corrected METs, the default collection to use
        ##

        jetsEnShift = jetsEnShiftForRawMEt.clone(
            addResidualJES = cms.bool(False)
            )

        ## Up Variation
        jetCollectionEnShift =addModuleToSequence(process, jetsEnShift,
                                           [ "shifted", jetCollection, "En"+ var ],
                                                  sequence, postfix)
        shiftedParticleCollections['jetCollectionEn' + var ] = jetCollectionEnShift
        collectionsToKeep.append(jetCollectionEnShift)


    return (shiftedParticleCollections, collectionsToKeep)



def addSmearedJets(process, jetCollection, smearedJetCollectionName_parts,
                   jetSmearFileName, jetSmearHistogram, jetResolutions,
                   varyByNsigmas, shiftBy = None, sequence = None, postfix = ""):

    smearedJets = cms.EDProducer("SmearedPATJetProducer",
            src = cms.InputTag(jetCollection),
            dRmaxGenJetMatch = cms.string('min(0.5, 0.1 + 0.3*exp(-0.05*(genJetPt - 10.)))'),
            sigmaMaxGenJetMatch = cms.double(3.),
            inputFileName = cms.FileInPath(jetSmearFileName),
            lutName = cms.string(jetSmearHistogram),
            jetResolutions = jetResolutions.METSignificance_params,
            # CV: skip jet smearing for pat::Jets for which the jet-energy correction (JEC) factors are either very large or negative
            #     since both cases produce unphysically large tails in the Type 1 corrected MET distribution after the smearing,
            #
            #     e.g. raw jet:   energy = 50 GeV, eta = 2.86, pt =  1   GeV
            #          corr. jet: energy = -3 GeV            , pt = -0.1 GeV (JEC factor L1fastjet*L2*L3 = -17)
            #                     energy = 10 GeV for corrected jet after smearing
            #         --> smeared raw jet energy = -170 GeV !!
            #
            #         --> (corr. - raw) jet contribution to MET = -1 (-10) GeV before (after) smearing,
            #             even though jet energy got smeared by merely 1 GeV
            #
            skipJetSelection = cms.string(
                'jecSetsAvailable & abs(energy - correctedP4("Uncorrected").energy) > (5.*min(energy, correctedP4("Uncorrected").energy))'
            ),
            skipRawJetPtThreshold = cms.double(10.), # GeV
            skipCorrJetPtThreshold = cms.double(1.e-2),
            verbosity = cms.int32(0)
        )
    if shiftBy is not None:
        setattr(smearedJets, "shiftBy", cms.double(shiftBy*varyByNsigmas))
    smearedJetCollection = addModuleToSequence(process, smearedJets,
                                               smearedJetCollectionName_parts,
                                               sequence, postfix)

    return smearedJetCollection

