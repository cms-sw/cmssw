import FWCore.ParameterSet.Config as cms


input_names = ["input_{}".format(i) for i in range(1,6)]
output_names = ["ID_pred/Softmax", "regression_pred/BiasAdd"]

pfDeepFlavourJetTags = cms.EDProducer(
    'DeepFlavourJetTagProducer',
    src = cms.InputTag('pfDeepFlavourTagInfos'),
    # path to .meta file (will be stripped)
    graph_path = cms.FileInPath('RecoBTag/Combined/data/DeepFlavourV01_C/saved_model.pb'),
    flav_table = cms.PSet(
                      probb = cms.vuint32([0]),
                      probbb = cms.vuint32([1]),
                      problepb = cms.vuint32([2]),
                      probc = cms.vuint32([3]),
                      probuds = cms.vuint32([4]),
                      probg = cms.vuint32([5]),
                      ),
    input_names = cms.vstring(input_names),
    output_names = cms.vstring(output_names),
    lp_names = cms.vstring('cpf_input_batchnorm/keras_learning_phase')
)
