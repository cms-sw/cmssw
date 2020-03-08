from RecoBTag.ONNXRuntime.SwitchProducerONNX import SwitchProducerONNX
from RecoBTag.ONNXRuntime.pfDeepFlavourONNXJetTags_cfi import pfDeepFlavourONNXJetTags
from RecoBTag.TensorFlow.pfDeepFlavourTFJetTags_cfi import pfDeepFlavourTFJetTags

pfDeepFlavourJetTags = SwitchProducerONNX(
    native = pfDeepFlavourTFJetTags.clone(),
    onnx = pfDeepFlavourONNXJetTags.clone(),
)
