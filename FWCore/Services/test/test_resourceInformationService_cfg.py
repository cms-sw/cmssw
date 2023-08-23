# Test ResourceInformationService
# The shell script will examine its printout
import FWCore.ParameterSet.Config as cms

# This class is a hack just for the test to get
# 'gpu-foo' into @selected_accelerators
class ProcessAcceleratorTest(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorTest,self).__init__()
        self._labels = ["test1", "gpu-foo", "test2"]
    def labels(self):
        return self._labels
    def enabledLabels(self):
        return self._labels

process = cms.Process("PROD")
process.add_(ProcessAcceleratorTest())

process.ResourceInformationService = cms.Service("ResourceInformationService",
  verbose = cms.untracked.bool(True)
)

process.options.accelerators = ["*"]

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

