import FWCore.ParameterSet.Config as cms


input_names = ["input_{}:0".format(i) for i in range(1,6)]
output_names = ["ID_pred/Softmax:0", "regression_pred/BiasAdd:0"]

pfDeepFlavourJetTags = cms.EDProducer(
    'DeepFlavourJetTagProducer',
    src = cms.InputTag('DeepFlavourTagInfos'),
    graph_path = cms.string('/afs/cern.ch/work/p/pdecastr/public/Deep/models_19072017/KERAS_model.h5_tfsession/tf'),
    outputs = cms.vstring(["probb"]),
    input_names = cms.vstring(input_names),
    output_names = cms.vstring(output_names)
)
