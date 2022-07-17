import FWCore.ParameterSet.Config as cms

def _switch_cuda(useAccelerators):
    have_gpu = ("gpu-nvidia" in useAccelerators)
    return (have_gpu, 2)

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

