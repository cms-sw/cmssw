from __future__ import division

from RecoHI.HiJetAlgos.HiRecoPFJets_cff import akCs4PFJets
from RecoHI.HiJetAlgos.HiGenJets_cff import ak5HiGenJets
from RecoHI.HiJetAlgos.HiGenCleaner_cff import heavyIonCleanedGenJets
from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonJets_cff import *
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

    from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection

    addJetCollection(
        process,
        labelName = 'AKCs4PF',
        jetSource = cms.InputTag('akCs4PFJets'),
        btagDiscriminators = [
            'simpleSecondaryVertexHighEffBJetTags',
            'simpleSecondaryVertexHighPurBJetTags',
            'combinedSecondaryVertexV2BJetTags',
            'jetBProbabilityBJetTags',
            'trackCountingHighEffBJetTags',
            'trackCountingHighPurBJetTags'
        ],
        genJetCollection = cms.InputTag('ak4HiCleanedGenJets'), 
        jetCorrections = ('AK4PF', ['L2Relative'], 'None'),
        getJetMCFlavour = False, # jet flavor disabled
    )

    process.patJetsAKCs4PF.addAssociatedTracks = True
    process.patJetsAKCs4PF.addBTagInfo = True
    process.patJetsAKCs4PF.useLegacyJetMCFlavour = True
    process.patJetCorrFactorsAKCs4PF.useNPV = False
    process.patJetCorrFactorsAKCs4PF.useRho = False
    process.patJetPartonMatchAKCs4PF.matched = "cleanedPartons"


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
