import FWCore.ParameterSet.Config as cms

hltDeepCombinedSecondaryVertexBJetTagsPFPuppiModEta2p4 = cms.EDProducer("DeepFlavourJetTagsProducer",
    NNConfig = cms.FileInPath('RecoBTag/Combined/data/DeepCSV_PhaseII.json'),
    checkSVForDefaults = cms.bool(True),
    meanPadding = cms.bool(True),
    src = cms.InputTag("hltDeepCombinedSecondaryVertexBJetTagsInfosPuppiModEta2p4"),
    toAdd = cms.PSet(
        probbb = cms.string('probb')
    )
)
