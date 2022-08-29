import FWCore.ParameterSet.Config as cms

class CreateGenMatch(object):
    def __init__(self,
            distance=0.3
            ):
        self.dR = distance

    def __call__(self, process, inputs):
        producer = process.l1tHGCal3DClusterGenMatchSelector.clone(
                dR = cms.double(self.dR),
                src = cms.InputTag(inputs)
                )
        return producer
