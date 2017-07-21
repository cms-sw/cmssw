import FWCore.ParameterSet.Config as cms


input_names = ["input_{}:0".format(i) for i in range(1,6)]
output_names = ["ID_pred/Softmax:0", "regression_pred/BiasAdd:0"]

pfDeepFlavourJetTags = cms.EDProducer(
    'DeepFlavourJetTagProducer',
    src = cms.InputTag('DeepFlavourTagInfos'),
    graph_path = cms.string('/afs/cern.ch/work/p/pdecastr/public/Deep/models_19072017/KERAS_model.h5_tfsession/tf'),
    flav_table = cms.PSet(
                      probb = cms.vuint32(0),
                      probbb = cms.vuint32(1),
                      problepb = cms.vuint32(2),
                      probc = cms.vuint32(3),
                      probuds = cms.vuint32(4),
                      probg = cms.vuint32(5),
                      ),
    input_names = cms.vstring(input_names),
    output_names = cms.vstring(output_names)
)
