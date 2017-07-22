import FWCore.ParameterSet.Config as cms


input_names = ["input_{}:0".format(i) for i in range(1,6)]
output_names = ["ID_pred/Softmax:0", "regression_pred/BiasAdd:0"]

pfDeepFlavourExporter = cms.EDAnalyzer(
    'DeepFlavourExporter',
    tag_info_src = cms.InputTag('pfDeepFlavourTagInfos'),
    jet_src = cms.InputTag('updatedPatJetsTransientCorrected'),
    btagDiscriminators = cms.vstring(
      'pfCombinedSecondaryVertexV2BJetTags',
      'pfDeepCSVJetTags:probb',
      'pfDeepFlavourJetTags:probb',
      'pfDeepFlavourJetTags:problepb',
      'pfDeepFlavourJetTags:probc',
      'pfDeepFlavourJetTags:probuds',
      'pfDeepFlavourJetTags:probg',
      ),
)
