from __future__ import division

import six

from RecoBTag.Configuration.RecoBTag_cff import *
from RecoBTag.SecondaryVertex.negativeCombinedSecondaryVertexV2BJetTags_cfi import *
from RecoBTag.SecondaryVertex.negativeCombinedSecondaryVertexV2Computer_cfi import *
from RecoBTag.SecondaryVertex.negativeSimpleSecondaryVertexHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.negativeSimpleSecondaryVertexHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.positiveCombinedSecondaryVertexV2BJetTags_cfi import *
from RecoBTag.SecondaryVertex.positiveCombinedSecondaryVertexV2Computer_cfi import *
from RecoBTag.SecondaryVertex.secondaryVertexNegativeTagInfos_cfi import *

from RecoHI.HiJetAlgos.HiRecoPFJets_cff import akCs4PFJets
from RecoHI.HiJetAlgos.HiGenJets_cff import ak5HiGenJets
from RecoHI.HiJetAlgos.HiGenCleaner_cff import heavyIonCleanedGenJets

from RecoJets.JetAssociationProducers.ak5JTA_cff import *

from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import *
from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonJets_cff import *
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import *
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
from PhysicsTools.PatAlgos.tools.helpers import *

def setupHeavyIonJets(process, tag, radius, task):
    addToProcessAndTask(
        tag + 'Jets',
        akCs4PFJets.clone(rParam = radius / 10),
        process,
        task)

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
        'JetTracksAssociatorAtVertex':
        ak5JetTracksAssociatorAtVertex.clone(
            jets = tag + "Jets",
            tracks = "highPurityTracks",
            ),

        'ImpactParameterTagInfos':
        impactParameterTagInfos.clone(
            jetTracks = "JetTracksAssociatorAtVertex" + tag,
            ),

        'TrackCountingHighEffBJetTags':
        trackCountingHighEffBJetTags.clone(
            tagInfos = ["ImpactParameterTagInfos" + tag],
            ),

        'TrackCountingHighPurBJetTags':
        trackCountingHighPurBJetTags.clone(
            tagInfos = ["ImpactParameterTagInfos" + tag],
            ),

        'JetProbabilityBJetTags':
        jetProbabilityBJetTags.clone(
            tagInfos = ["ImpactParameterTagInfos" + tag],
            ),

        'JetBProbabilityBJetTags':
        jetBProbabilityBJetTags.clone(
            tagInfos = ["ImpactParameterTagInfos" + tag],
            ),

        'SecondaryVertexTagInfos':
        secondaryVertexTagInfos.clone(
            trackIPTagInfos = "ImpactParameterTagInfos" + tag,
            ),

        'CombinedSecondaryVertexBJetTags':
        combinedSecondaryVertexV2BJetTags.clone(
            tagInfos = [
                "ImpactParameterTagInfos" + tag,
                "SecondaryVertexTagInfos" + tag,
                ],
            ),

        'CombinedSecondaryVertexV2BJetTags':
        combinedSecondaryVertexV2BJetTags.clone(
            tagInfos = [
                "ImpactParameterTagInfos" + tag,
                "SecondaryVertexTagInfos" + tag,
                ],
            ),

        'SecondaryVertexTagInfos':
        secondaryVertexTagInfos.clone(
            trackIPTagInfos = "ImpactParameterTagInfos" + tag,
            ),

        'SimpleSecondaryVertexHighEffBJetTags':
        simpleSecondaryVertexHighEffBJetTags.clone(
            tagInfos = ["SecondaryVertexTagInfos" + tag],
            ),

        'SimpleSecondaryVertexHighPurBJetTags':
        simpleSecondaryVertexHighPurBJetTags.clone(
            tagInfos = ["SecondaryVertexTagInfos" + tag],
            ),

        'CombinedSecondaryVertexBJetTags':
        combinedSecondaryVertexV2BJetTags.clone(
            tagInfos = [
                "ImpactParameterTagInfos" + tag,
                "SecondaryVertexTagInfos" + tag,
                ],
            ),

        'CombinedSecondaryVertexV2BJetTags':
        combinedSecondaryVertexV2BJetTags.clone(
            tagInfos = [
                "ImpactParameterTagInfos" + tag,
                "SecondaryVertexTagInfos" + tag,
                ],
            ),

        'SecondaryVertexNegativeTagInfos':
        secondaryVertexNegativeTagInfos.clone(
            trackIPTagInfos = "ImpactParameterTagInfos" + tag,
            ),

        'NegativeSimpleSecondaryVertexHighEffBJetTags':
        negativeSimpleSecondaryVertexHighEffBJetTags.clone(
            tagInfos = ["SecondaryVertexNegativeTagInfos" + tag],
            ),

        'NegativeSimpleSecondaryVertexHighPurBJetTags':
        negativeSimpleSecondaryVertexHighPurBJetTags.clone(
            tagInfos = ["SecondaryVertexNegativeTagInfos" + tag],
            ),

        'NegativeCombinedSecondaryVertexBJetTags':
        negativeCombinedSecondaryVertexV2BJetTags.clone(
            tagInfos = [
                "ImpactParameterTagInfos" + tag,
                "SecondaryVertexNegativeTagInfos" + tag,
                ],
            ),

        'PositiveCombinedSecondaryVertexBJetTags':
        positiveCombinedSecondaryVertexV2BJetTags.clone(
            tagInfos = [
                "ImpactParameterTagInfos" + tag,
                "SecondaryVertexTagInfos" + tag,
                ],
            ),

        'NegativeCombinedSecondaryVertexV2BJetTags':
        negativeCombinedSecondaryVertexV2BJetTags.clone(
            tagInfos = [
                "ImpactParameterTagInfos" + tag,
                "SecondaryVertexNegativeTagInfos" + tag,
                ],
            ),

        'PositiveCombinedSecondaryVertexV2BJetTags':
        positiveCombinedSecondaryVertexV2BJetTags.clone(
            tagInfos = [
                "ImpactParameterTagInfos" + tag,
                "SecondaryVertexTagInfos" + tag,
                ],
            ),

        'SoftPFMuonsTagInfos':
        softPFMuonsTagInfos.clone(
            jets = tag + "Jets",
            ),

        'SoftPFMuonBJetTags':
        softPFMuonBJetTags.clone(
            tagInfos = ["SoftPFMuonsTagInfos" + tag],
            ),

        'SoftPFMuonByIP3dBJetTags':
        softPFMuonByIP3dBJetTags.clone(
            tagInfos = ["SoftPFMuonsTagInfos" + tag],
            ),

        'SoftPFMuonByPtBJetTags':
        softPFMuonByPtBJetTags.clone(
            tagInfos = ["SoftPFMuonsTagInfos" + tag],
            ),

        'PositiveSoftPFMuonByPtBJetTags':
        positiveSoftPFMuonByPtBJetTags.clone(
            tagInfos = ["SoftPFMuonsTagInfos" + tag],
            ),

        'NegativeSoftPFMuonByPtBJetTags':
        negativeSoftPFMuonByPtBJetTags.clone(
            tagInfos = ["SoftPFMuonsTagInfos" + tag],
            ),

        'patJetCorrFactors':
        patJetCorrFactors.clone(
            useNPV = False,
            useRho = False,
            levels = ['L2Relative'],
            payload = "AK" + str(radius) + "PF",
            src = tag + "Jets",
            ),

        'patJetGenJetMatch':
        patJetGenJetMatch.clone(
            matched = 'ak' + str(radius) +  "HiCleanedGenJets",
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

        'patJets':
        patJets.clone(
            jetSource = tag + "Jets",
            genJetMatch = "patJetGenJetMatch" + tag,
            genPartonMatch = "patJetPartonMatch" + tag,
            JetFlavourInfoSource = "patJetFlavourAssociation" + tag,
            JetPartonMapSource = "patJetFlavourAssociationLegacy" + tag,
            jetCorrFactorsSource = ["patJetCorrFactors" + tag],
            trackAssociationSource = "JetTracksAssociatorAtVertex" + tag,
            useLegacyJetMCFlavour = True,
            discriminatorSources = [
                "SimpleSecondaryVertexHighEffBJetTags" + tag,
                "SimpleSecondaryVertexHighPurBJetTags" + tag,
                "CombinedSecondaryVertexBJetTags" + tag,
                "CombinedSecondaryVertexV2BJetTags" + tag,
                "JetBProbabilityBJetTags" + tag,
                "JetProbabilityBJetTags" + tag,
                "TrackCountingHighEffBJetTags" + tag,
                "TrackCountingHighPurBJetTags" + tag,
                ],
            tagInfoSources = [
                "ImpactParameterTagInfos" + tag,
                "SecondaryVertexTagInfos" + tag,
                ],
            addJetCharge = False,
            addTagInfos = False,
            ),
        }

    for label, module in six.iteritems(modules):
        addToProcessAndTask(label + tag, module, process, task)

def aliasCsJets(process, tag):
    delattr(process, 'patJets')

    source = cms.VPSet(cms.PSet(type = cms.string('patJets')))
    process.patJets = cms.EDAlias(**{ 'patJets' + tag: source })

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
    process.basicJetsForMetPuppi.jetCorrLabelRes = 'Uncorrected'

def addJECsForData(process):
    for label in process.producerNames().split():
        module = getattr(process, label)
        if module.type_() == "JetCorrFactorsProducer":
            module.levels.append('L2L3Residual')
