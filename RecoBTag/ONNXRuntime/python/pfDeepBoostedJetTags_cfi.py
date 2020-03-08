from RecoBTag.ONNXRuntime.SwitchProducerONNX import SwitchProducerONNX
from RecoBTag.ONNXRuntime.deepBoostedJetONNXJetTagsProducer_cfi import deepBoostedJetONNXJetTagsProducer
from RecoBTag.MXNet.boostedJetMXNetJetTagsProducer_cfi import boostedJetMXNetJetTagsProducer
from RecoBTag.ONNXRuntime.Parameters.DeepBoostedJet.V02.pfDeepBoostedJetPreprocessParams_cfi import pfDeepBoostedJetPreprocessParams
from RecoBTag.ONNXRuntime.Parameters.DeepBoostedJet.V02.pfMassDecorrelatedDeepBoostedJetPreprocessParams_cfi import pfMassDecorrelatedDeepBoostedJetPreprocessParams

_flav_names = ['probTbcq', 'probTbqq', 'probTbc', 'probTbq', 'probWcq', 'probWqq', 
               'probZbb', 'probZcc', 'probZqq', 'probHbb', 'probHcc', 'probHqqqq', 
               'probQCDbb', 'probQCDcc', 'probQCDb', 'probQCDc', 'probQCDothers']

# nominal DeepAK8
pfDeepBoostedJetTags = SwitchProducerONNX(
    native = boostedJetMXNetJetTagsProducer.clone(
        flav_names = _flav_names,
        preprocessParams = pfDeepBoostedJetPreprocessParams,
        model_path = 'RecoBTag/Combined/data/DeepBoostedJet/V02/full/resnet-symbol.json',
        param_path = 'RecoBTag/Combined/data/DeepBoostedJet/V02/full/resnet-0000.params',
        ),
    onnx = deepBoostedJetONNXJetTagsProducer.clone(
        flav_names = _flav_names,
        preprocessParams = pfDeepBoostedJetPreprocessParams,
        model_path = 'RecoBTag/Combined/data/DeepBoostedJet/V02/full/resnet.onnx',
        ),
    )

# mass-decorrelated DeepAK8
pfMassDecorrelatedDeepBoostedJetTags = SwitchProducerONNX(
    native = boostedJetMXNetJetTagsProducer.clone(
        flav_names = _flav_names,
        preprocessParams = pfMassDecorrelatedDeepBoostedJetPreprocessParams,
        model_path = 'RecoBTag/Combined/data/DeepBoostedJet/V02/decorrelated/resnet-symbol.json',
        param_path = 'RecoBTag/Combined/data/DeepBoostedJet/V02/decorrelated/resnet-0000.params',
        ),
    onnx = deepBoostedJetONNXJetTagsProducer.clone(
        flav_names = _flav_names,
        preprocessParams = pfMassDecorrelatedDeepBoostedJetPreprocessParams,
        model_path = 'RecoBTag/Combined/data/DeepBoostedJet/V02/decorrelated/resnet.onnx',
        ),
    )

