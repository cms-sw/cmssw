from __future__ import division

import six

from RecoHI.HiJetAlgos.HiRecoPFJets_cff import akCs4PFJets
from RecoHI.HiJetAlgos.HiGenJets_cff import ak5HiGenJets
from RecoHI.HiJetAlgos.HiGenCleaner_cff import heavyIonCleanedGenJets
from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonJets_cff import *
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import *
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import *
from PhysicsTools.PatAlgos.tools.helpers import *

from RecoBTag.Configuration.RecoBTag_cff import *
from RecoJets.JetAssociationProducers.ak5JTA_cff import *

def setupHeavyIonJets(process, tag, radius, task):
    addToProcessAndTask(
        tag + 'Jets',
        akCs4PFJets.clone(rParam = radius / 10),
        process,
        task)

    modules = {
        'jetTracksAssociatorAtVertex':
        ak5JetTracksAssociatorAtVertex.clone(
            jets = tag + "Jets",
            tracks = "highPurityGeneralTracks",
            ),

        'impactParameterTagInfos':
        impactParameterTagInfos.clone(
            jetTracks = "jetTracksAssociatorAtVertex" + tag,
            ),

        'trackCountingHighEffBJetTags':
        trackCountingHighEffBJetTags.clone(
            tagInfos = ["impactParameterTagInfos" + tag],
            ),

        'trackCountingHighPurBJetTags':
        trackCountingHighPurBJetTags.clone(
            tagInfos = ["impactParameterTagInfos" + tag],
            ),

        'jetProbabilityBJetTags':
        jetProbabilityBJetTags.clone(
            tagInfos = ["impactParameterTagInfos" + tag],
            ),

        'jetBProbabilityBJetTags':
        jetBProbabilityBJetTags.clone(
            tagInfos = ["impactParameterTagInfos" + tag],
            ),

        'secondaryVertexTagInfos':
        secondaryVertexTagInfos.clone(
            trackIPTagInfos = "impactParameterTagInfos" + tag,
            ),

        'combinedSecondaryVertexBJetTags':
        combinedSecondaryVertexV2BJetTags.clone(
            tagInfos = [
                "impactParameterTagInfos" + tag,
                "secondaryVertexTagInfos" + tag,
                ],
            ),

        'combinedSecondaryVertexV2BJetTags':
        combinedSecondaryVertexV2BJetTags.clone(
            tagInfos = [
                "impactParameterTagInfos" + tag,
                "secondaryVertexTagInfos" + tag,
                ],
            ),

        'secondaryVertexTagInfos':
        secondaryVertexTagInfos.clone(
            trackIPTagInfos = "impactParameterTagInfos" + tag,
            ),

        'simpleSecondaryVertexHighEffBJetTags':
        simpleSecondaryVertexHighEffBJetTags.clone(
            tagInfos = ["secondaryVertexTagInfos" + tag],
            ),

        'simpleSecondaryVertexHighPurBJetTags':
        simpleSecondaryVertexHighPurBJetTags.clone(
            tagInfos = ["secondaryVertexTagInfos" + tag],
            ),

        'combinedSecondaryVertexBJetTags':
        combinedSecondaryVertexV2BJetTags.clone(
            tagInfos = [
                "impactParameterTagInfos" + tag,
                "secondaryVertexTagInfos" + tag,
                ],
            ),

        'combinedSecondaryVertexV2BJetTags':
        combinedSecondaryVertexV2BJetTags.clone(
            tagInfos = [
                "impactParameterTagInfos" + tag,
                "secondaryVertexTagInfos" + tag,
                ],
            ),

        'patJetCorrFactors':
        patJetCorrFactors.clone(
            useNPV = False,
            useRho = False,
            levels = ['L2Relative'],
            payload = "AK" + str(radius) + "PF",
            src = tag + "Jets",
            ),


        }

    process.load("RecoBTag.ImpactParameter.impactParameter_EventSetup_cff")
    process.load("RecoBTag.SecondaryVertex.secondaryVertex_EventSetup_cff")
    for label, module in six.iteritems(modules):
        addToProcessAndTask(label + tag, module, process, task)

def setupHeavyIonGenJets(process, tag, radius, task):
    genjetcollection = 'ak' + str(radius) + 'HiGenJets'

    addToProcessAndTask(
        genjetcollection,
        ak5HiGenJets.clone(rParam = radius / 10),
        process,
        task)

    addToProcessAndTask(
        'ak' + str(radius) + 'HiCleanedGenJets',
        heavyIonCleanedGenJets.clone(src = genjetcollection),
        process,
        task)

    modules = {
        'patJetGenJetMatch':
        patJetGenJetMatch.clone(
            matched = 'ak' + str(radius) + 'HiCleanedGenJets',
            maxDeltaR = radius / 10,
            resolveByMatchQuality = True,
            src = tag + "Jets",
            ),

        'patJetPartonMatch':
        patJetPartonMatch.clone(
            matched = "cleanedPartons",
            src = tag + "Jets",
            ),

        'patJetPartons':
        patJetPartons.clone(
            particles = "hiSignalGenParticles",
            ),

        'patJetFlavourAssociation':
        patJetFlavourAssociation.clone(
            jets = tag + "Jets",
            rParam = radius / 10,
            bHadrons = "patJetPartons" + tag + ":bHadrons",
            cHadrons = "patJetPartons" + tag + ":cHadrons",
            leptons = "patJetPartons" + tag + ":leptons",
            partons = "patJetPartons" + tag + ":physicsPartons",
            ),

        'patJetPartonAssociationLegacy':
        patJetPartonAssociationLegacy.clone(
            jets = tag + "Jets",
            partons = "allPartons",
            ),

        'patJetFlavourAssociationLegacy':
        patJetFlavourAssociationLegacy.clone(
            srcByReference = "patJetPartonAssociationLegacy" + tag,
            ),
        }

    for label, module in six.iteritems(modules):
        addToProcessAndTask(label + tag, module, process, task)

def removeL1FastJetJECs(process):
    for label in process.producerNames().split():
        module = getattr(process, label)
        if module.type_() == "PATPFJetMETcorrInputProducer":
            module.offsetCorrLabel = ''

def removeJECsForMC(process):
    for label in process.producerNames().split():
        module = getattr(process, label)
        if module.type_() == "PATPFJetMETcorrInputProducer":
            module.jetCorrLabel = 'Uncorrected'

    process.basicJetsForMet.jetCorrLabel = 'Uncorrected'

def addJECsForData(process):
    for label in process.producerNames().split():
        module = getattr(process, label)
        if module.type_() == "JetCorrFactorsProducer":
            module.levels.append('L2L3Residual')
