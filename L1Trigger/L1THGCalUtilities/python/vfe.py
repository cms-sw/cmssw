import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.l1tHGCalVFEProducer_cfi import vfe_proc

class CreateVfe(object):
    def __init__(self,
            linearization_si=vfe_proc.linearizationCfg_si,
            linearization_sc=vfe_proc.linearizationCfg_sc,
            compression_ldm=vfe_proc.compressionCfg_ldm,
            compression_hdm=vfe_proc.compressionCfg_hdm,
            ):
        self.processor = vfe_proc.clone(
            linearizationCfg_si = linearization_si,
            linearizationCfg_sc = linearization_sc,
            compressionCfg_ldm = compression_ldm,
            compressionCfg_hdm = compression_hdm,
        )

    def __call__(self, process):
        producer = process.l1tHGCalVFEProducer.clone(
            ProcessorParameters = self.processor
        )
        return producer
