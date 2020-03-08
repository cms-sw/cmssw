from RecoBTag.ONNXRuntime.SwitchProducerONNX import SwitchProducerONNX
from RecoBTag.ONNXRuntime.pfDeepDoubleBvLONNXJetTags_cfi import pfDeepDoubleBvLONNXJetTags
from RecoBTag.ONNXRuntime.pfDeepDoubleCvLONNXJetTags_cfi import pfDeepDoubleCvLONNXJetTags
from RecoBTag.ONNXRuntime.pfDeepDoubleCvBONNXJetTags_cfi import pfDeepDoubleCvBONNXJetTags
from RecoBTag.TensorFlow.pfDeepDoubleBvLTFJetTags_cfi import pfDeepDoubleBvLTFJetTags
from RecoBTag.TensorFlow.pfDeepDoubleCvLTFJetTags_cfi import pfDeepDoubleCvLTFJetTags
from RecoBTag.TensorFlow.pfDeepDoubleCvBTFJetTags_cfi import pfDeepDoubleCvBTFJetTags

pfDeepDoubleBvLJetTags = SwitchProducerONNX(
    native = pfDeepDoubleBvLTFJetTags.clone(),
    onnx = pfDeepDoubleBvLONNXJetTags.clone(),
)

pfDeepDoubleCvLJetTags = SwitchProducerONNX(
    native = pfDeepDoubleCvLTFJetTags.clone(),
    onnx = pfDeepDoubleCvLONNXJetTags.clone(),
)

pfDeepDoubleCvBJetTags = SwitchProducerONNX(
    native = pfDeepDoubleCvBTFJetTags.clone(),
    onnx = pfDeepDoubleCvBONNXJetTags.clone(),
)

pfMassIndependentDeepDoubleBvLJetTags = SwitchProducerONNX(
    native = pfDeepDoubleBvLTFJetTags.clone(
        model_path = 'RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDB_mass_independent.pb'
        ),
    onnx = pfDeepDoubleBvLONNXJetTags.clone(
        model_path = 'RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDB_mass_independent.onnx'
        ),
)

pfMassIndependentDeepDoubleCvLJetTags = SwitchProducerONNX(
    native = pfDeepDoubleCvLTFJetTags.clone(
        model_path = 'RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDC_mass_independent.pb'
        ),
    onnx = pfDeepDoubleCvLONNXJetTags.clone(
        model_path = 'RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDC_mass_independent.onnx'
        ),
)

pfMassIndependentDeepDoubleCvBJetTags = SwitchProducerONNX(
    native = pfDeepDoubleCvBTFJetTags.clone(
        model_path = 'RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDCvB_mass_independent.pb'
        ),
    onnx = pfDeepDoubleCvBONNXJetTags.clone(
        model_path = 'RecoBTag/Combined/data/DeepDoubleX/94X/V01/DDCvB_mass_independent.onnx'
        ),
)
