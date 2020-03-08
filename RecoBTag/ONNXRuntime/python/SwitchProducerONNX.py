import FWCore.ParameterSet.Config as cms

_tf_enabled_cached = None
_onnxrt_enabled_cached = None


def _switch_native():
    global _tf_enabled_cached
    if _tf_enabled_cached is None:
        import os
        _tf_enabled_cached = ('CMS_DISABLE_TENSORFLOW' not in os.environ)
    return (_tf_enabled_cached, 1)


def _switch_onnxruntime():
    global _onnxrt_enabled_cached
    if _onnxrt_enabled_cached is None:
        import os
        _onnxrt_enabled_cached = ('amd64' in os.environ['SCRAM_ARCH'] or 'aarch64' in os.environ['SCRAM_ARCH']) and ('CMS_DISABLE_ONNXRUNTIME' not in os.environ)
    return (_onnxrt_enabled_cached, 2)


class SwitchProducerONNX(cms.SwitchProducer):

    def __init__(self, **kwargs):
        super(SwitchProducerONNX, self).__init__(
            dict(native = _switch_native,
                 onnx = _switch_onnxruntime),
            **kwargs
        )

    def cloneAll(self, **params):
        return super(SwitchProducerONNX, self).clone(
            native = self.native.clone(**params),
            onnx = self.onnx.clone(**params),
            )


cms.specialImportRegistry.registerSpecialImportForType(SwitchProducerONNX, "from RecoBTag.ONNXRuntime.SwitchProducerONNX import SwitchProducerONNX")
