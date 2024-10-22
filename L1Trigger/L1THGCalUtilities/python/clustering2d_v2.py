import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCal.l1tHGCalBackEndLayer1Producer_cfi import dummy_C2d_params, \
                                                              stage1truncation_proc, \
                                                              truncation_params

class RozBinTruncation(object):
    def __init__(self,
            maxTcsPerBin=truncation_params.maxTcsPerBin,
            doTruncation=truncation_params.doTruncation):
        self.processor = stage1truncation_proc.clone(
                truncation_parameters=truncation_params.clone(
                maxTcsPerBin=maxTcsPerBin,
                doTruncation=doTruncation
                )
        )

    def __call__(self,process,inputs):
        producer = process.l1tHGCalBackEndStage1Producer.clone(
            InputTriggerCells = cms.InputTag(inputs),
            C2d_parameters = dummy_C2d_params.clone(),
            ProcessorParameters = self.processor
        )
        return producer
