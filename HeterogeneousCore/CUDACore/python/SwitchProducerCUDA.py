import FWCore.ParameterSet.Config as cms

_cuda_enabled_cached = None

def _switch_cuda():
    global _cuda_enabled_cached
    if _cuda_enabled_cached is None:
        import os
        _cuda_enabled_cached = (os.system("cudaIsEnabled") == 0)
    return (_cuda_enabled_cached, 2)

class SwitchProducerCUDA(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerCUDA,self).__init__(
            dict(cpu = cms.SwitchProducer.getCpu(),
                 cuda = _switch_cuda),
            **kargs
        )
cms.specialImportRegistry.registerSpecialImportForType(SwitchProducerCUDA, "from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA")

if __name__ == "__main__":
    import unittest

    class TestSwitchProducerCUDA(unittest.TestCase):
        def testPickle(self):
            import pickle
            sp = SwitchProducerCUDA(cpu = cms.EDProducer("Foo"), cuda = cms.EDProducer("Bar"))
            pkl = pickle.dumps(sp)
            unpkl = pickle.loads(pkl)
            self.assertEqual(unpkl.cpu.type_(), "Foo")
            self.assertEqual(unpkl.cuda.type_(), "Bar")

    unittest.main()

