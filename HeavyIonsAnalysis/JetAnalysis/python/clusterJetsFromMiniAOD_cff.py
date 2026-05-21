from __future__ import division

from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import *
from RecoJets.Configuration.RecoGenJets_cff import ak4GenJetsNoNu
from RecoBTag.ImpactParameter.pfImpactParameterTagInfos_cfi import pfImpactParameterTagInfos
from RecoBTag.SecondaryVertex.pfSecondaryVertexTagInfos_cfi import pfSecondaryVertexTagInfos
from RecoBTag.ImpactParameter.pfJetProbabilityBJetTags_cfi import pfJetProbabilityBJetTags
from RecoBTag.Combined.pfDeepCSVTagInfos_cfi import pfDeepCSVTagInfos
from RecoBTag.Combined.pfDeepCSVJetTags_cfi import pfDeepCSVJetTags

def get_radius(tag):
    return int("".join(filter(str.isdigit, tag)))

def addToSequence(label, module, process, sequence):
    setattr(process, label, module)
    sequence += getattr(process, label)

def setupHeavyIonJets(tag, sequence, process, isMC, radius = -1, JECTag = 'None', doFlow = False, matchJets = False):

    if radius < 0:
       radiustag = get_radius(tag)
       radius = radiustag * 0.1
    else:
       radiustag = "{:.0f}".format(radius*10)

    addToSequence( tag+'Jets',
                   akCs4PFJets.clone(rParam = radius, src = 'packedPFCandidates', useModulatedRho = doFlow),
                   process, sequence)

    if JECTag == 'None':
       JECTag = 'AK' + str(radiustag) + 'PF'

    addToSequence( tag+'patJetCorrFactors',
                   patJetCorrFactors.clone(payload = JECTag, src = tag+'Jets'),
                   process, sequence)

    # Add Monte Carlo specific collections to the sequence
    if isMC :

        genjetcollection = 'ak'+str(radiustag)+'GenJetsNoNu'

        addToSequence( genjetcollection,
                       ak4GenJetsNoNu.clone(src = 'packedGenParticlesSignal', rParam = radius),
                       process, sequence)

        # To find parton flavor for CS subtracted jets, they need to be matched with unsubtracted jets
        if matchJets:

            unsubtractedJetTag = "ak" + str(radiustag) + "PFMatchingFor" + tag

            addToSequence( unsubtractedJetTag+'Jets',
                           ak4PFJets.clone(rParam = radius, src = 'packedPFCandidates'),
                           process, sequence)

            addToSequence( unsubtractedJetTag+'patJetCorrFactors',
                           patJetCorrFactors.clone(payload = JECTag, src = unsubtractedJetTag+'Jets'),
                           process, sequence)

            addToSequence( unsubtractedJetTag+'patJetPartonMatch',
                           patJetPartonMatch.clone(maxDeltaR = radius,
                               matched = 'hiSignalGenParticles',
                               src = unsubtractedJetTag+'Jets'),
                           process, sequence)

            addToSequence( unsubtractedJetTag+'patJetGenJetMatch',
                           patJetGenJetMatch.clone(maxDeltaR = radius,
                               matched = genjetcollection,
                               src = unsubtractedJetTag + 'Jets'),
                           process, sequence)

            addToSequence( unsubtractedJetTag+'patJetPartons',
                           patJetPartons.clone(partonMode = 'Pythia8'),
                           process, sequence)

            addToSequence( unsubtractedJetTag+'patJetFlavourAssociation',
                           patJetFlavourAssociation.clone(jets = unsubtractedJetTag+'Jets',
                               rParam = radius,
                               bHadrons = cms.InputTag(unsubtractedJetTag+"patJetPartons","bHadrons"),
                               cHadrons = cms.InputTag(unsubtractedJetTag+"patJetPartons","cHadrons"),
                               partons = cms.InputTag(unsubtractedJetTag+"patJetPartons","physicsPartons"),
                               leptons = cms.InputTag(unsubtractedJetTag+"patJetPartons","leptons")),
                           process, sequence)

            addToSequence( unsubtractedJetTag+'patJets',
                           patJets.clone(
                               JetFlavourInfoSource = unsubtractedJetTag+'patJetFlavourAssociation',
                               JetPartonMapSource = unsubtractedJetTag+'patJetFlavourAssociation',
                               genJetMatch = unsubtractedJetTag+'patJetGenJetMatch',
                               genPartonMatch = unsubtractedJetTag+'patJetPartonMatch',
                               jetCorrFactorsSource = cms.VInputTag(unsubtractedJetTag+'patJetCorrFactors'),
                               jetSource = unsubtractedJetTag+'Jets',
                               addBTagInfo = False,
                               addDiscriminators = False,
                               addAssociatedTracks = False,
                               useLegacyJetMCFlavour = False
                           ),
                         process, sequence)

        # Configuration for unsubtracted jets ready

        addToSequence( tag+'patJetPartonMatch',
                       patJetPartonMatch.clone(maxDeltaR = radius,
                       matched = 'hiSignalGenParticles',
                       src = tag+'Jets'),
                       process, sequence)

        addToSequence( tag+'patJetGenJetMatch',
                       patJetGenJetMatch.clone(maxDeltaR = radius,
                          matched = genjetcollection,
                          src = tag + 'Jets'),
                       process, sequence)

        addToSequence( tag+'patJetPartonAssociationLegacy',
                       patJetPartonAssociationLegacy.clone(jets = tag + 'Jets'),
                       process, sequence)

        addToSequence( tag+'patJetFlavourAssociationLegacy',
                       patJetFlavourAssociationLegacy.clone(srcByReference = tag+'patJetPartonAssociationLegacy'),
                       process, sequence)

        addToSequence( tag+'patJetPartons',
                       patJetPartons.clone(partonMode = 'Pythia8'),
                       process, sequence)

    # Monte Carlo specific collections added

    addToSequence( tag+'pfImpactParameterTagInfos',
                   pfImpactParameterTagInfos.clone(jets = tag +'Jets',
                      candidates = 'packedPFCandidates', primaryVertex = 'offlineSlimmedPrimaryVertices'),
                   process, sequence)

    addToSequence( tag+'pfSecondaryVertexTagInfos',
                   pfSecondaryVertexTagInfos.clone(trackIPTagInfos = tag+'pfImpactParameterTagInfos'),
                   process, sequence)

    addToSequence( tag+'pfDeepCSVTagInfos',
                   pfDeepCSVTagInfos.clone(svTagInfos = tag+'pfSecondaryVertexTagInfos'),
                   process, sequence)

    addToSequence( tag+'pfDeepCSVJetTags',
                   pfDeepCSVJetTags.clone(src = tag+'pfDeepCSVTagInfos'),
                   process, sequence)

    addToSequence( tag+'pfJetProbabilityBJetTags',
                   pfJetProbabilityBJetTags.clone(tagInfos = [tag+'pfImpactParameterTagInfos']),
                   process, sequence)

    addToSequence( tag+'patJets',
                   patJets.clone(
                       JetFlavourInfoSource = tag+'patJetFlavourAssociation',
                       JetPartonMapSource = tag+'patJetFlavourAssociationLegacy',
                       genJetMatch = tag+'patJetGenJetMatch',
                       genPartonMatch = tag+'patJetPartonMatch',
                       jetCorrFactorsSource = cms.VInputTag(tag+'patJetCorrFactors'),
                       jetSource = tag+'Jets',
                       discriminatorSources = cms.VInputTag(cms.InputTag(tag+'pfDeepCSVJetTags','probb'), cms.InputTag(tag+'pfDeepCSVJetTags','probc'), cms.InputTag(tag+'pfDeepCSVJetTags','probudsg'), cms.InputTag(tag+'pfDeepCSVJetTags','probbb'), cms.InputTag(tag+'pfJetProbabilityBJetTags')),
                       addAssociatedTracks = False,
                   ),
                   process, sequence)

def setupPprefJets(tag, sequence, process, isMC, radius = -1, JECTag = 'None'):

    if radius < 0:
       radiustag = get_radius(tag)
       radius = radiustag * 0.1
    else:
       radiustag = get_radius(tag)

    addToSequence( tag+'Jets',
                   ak4PFJets.clone(rParam = radius, src = 'packedPFCandidates'),
                   process, sequence)

    if JECTag == 'None':
       JECTag = 'AK' + str(radiustag) + 'PF'

    addToSequence( tag+'patJetCorrFactors',
                   patJetCorrFactors.clone(payload = JECTag, src = tag+'Jets'),
                   process, sequence)

    if isMC :
        addToSequence( tag+'patJetPartonMatch',
                       patJetPartonMatch.clone(maxDeltaR = radius,
                          matched = 'hiSignalGenParticles',
                          src = tag+'Jets'),
                       process, sequence)

        genjetcollection = 'ak'+str(radiustag)+'GenJetsNoNu'

        addToSequence( genjetcollection,
                       ak4GenJetsNoNu.clone(src = 'packedGenParticles', rParam = radius),
                       process, sequence)

        addToSequence( tag+'patJetGenJetMatch',
                       patJetGenJetMatch.clone(maxDeltaR = radius,
                          matched = genjetcollection,
                          src = tag + 'Jets'),
                       process, sequence)

        addToSequence( tag+'patJetPartonAssociationLegacy',
                       patJetPartonAssociationLegacy.clone(jets = tag + 'Jets', partons = "allPartons"),
                       process, sequence)

        addToSequence( tag+'patJetFlavourAssociationLegacy',
                       patJetFlavourAssociationLegacy.clone(srcByReference = tag+'patJetPartonAssociationLegacy'),
                       process, sequence)

        addToSequence( tag+'patJetPartons',
                       patJetPartons.clone(particles = "hiSignalGenParticles" ,partonMode = 'Pythia8'),
                       process, sequence)

    addToSequence( tag+'pfImpactParameterTagInfos',
                   pfImpactParameterTagInfos.clone(jets = tag +'Jets',
                      candidates = 'packedPFCandidates', primaryVertex = 'offlineSlimmedPrimaryVertices'),
                   process, sequence)

    addToSequence( tag+'pfSecondaryVertexTagInfos',
                   pfSecondaryVertexTagInfos.clone(trackIPTagInfos = tag+'pfImpactParameterTagInfos'),
                   process, sequence)

    addToSequence( tag+'pfDeepCSVTagInfos',
                   pfDeepCSVTagInfos.clone(svTagInfos = tag+'pfSecondaryVertexTagInfos'),
                   process, sequence)

    addToSequence( tag+'pfDeepCSVJetTags',
                   pfDeepCSVJetTags.clone(src = tag+'pfDeepCSVTagInfos'),
                   process, sequence)

    addToSequence( tag+'pfJetProbabilityBJetTags',
                   pfJetProbabilityBJetTags.clone(tagInfos = [tag+'pfImpactParameterTagInfos']),
                   process, sequence)

    addToSequence( tag+'patJets',
                   patJets.clone(
                       jetSource = tag+'Jets',
                       genJetMatch = tag+'patJetGenJetMatch',
                       genPartonMatch = tag+'patJetPartonMatch',
                       JetFlavourInfoSource = tag+'patJetFlavourAssociation',
                       JetPartonMapSource = tag+'patJetFlavourAssociationLegacy',
                       jetCorrFactorsSource = cms.VInputTag(tag+'patJetCorrFactors'),
                       trackAssociationSource = "",
                       useLegacyJetMCFlavour = True,
                       discriminatorSources = cms.VInputTag(cms.InputTag(tag+'pfDeepCSVJetTags','probb'), cms.InputTag(tag+'pfDeepCSVJetTags','probc'), cms.InputTag(tag+'pfDeepCSVJetTags','probudsg'), cms.InputTag(tag+'pfDeepCSVJetTags','probbb'), cms.InputTag(tag+'pfJetProbabilityBJetTags')),
                       tagInfoSources = [],
                       addJetCharge = False,
                       addTagInfos = False,
                       addDiscriminators = False,
                       addAssociatedTracks = False
                   ),
                   process, sequence)
