import FWCore.ParameterSet.Config as cms

_thresholdsHB = cms.vdouble(0.8, 0.8, 0.8, 0.8)
_thresholdsHE = cms.vdouble(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
_thresholdsHBphase1 = cms.vdouble(0.1, 0.2, 0.3, 0.3)
_thresholdsHEphase1 = cms.vdouble(0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)

_module_pset = cms.PSet(
    navigator = cms.PSet(
            name = cms.string("PFRecHitHCALDenseIdNavigator"),
            hcalEnums = cms.vint32(1,2)
    ),
    producers = cms.VPSet(
           cms.PSet(
             name = cms.string("PFHBHERecHitCreator"),
             src = cms.InputTag("hbhereco",""),
             qualityTests = cms.VPSet(
                  cms.PSet(
                  threshold = cms.double(0.8), # only needed by GPU, seems redundant - to be investigated    
                  name = cms.string("PFRecHitQTestHCALThresholdVsDepth"),
                  cuts = cms.VPSet(
                        cms.PSet(
                            depth=cms.vint32(1, 2, 3, 4),
                            threshold = _thresholdsHB,
                            detectorEnum = cms.int32(1)
                            ),
                        cms.PSet(
                            depth=cms.vint32(1, 2, 3, 4, 5, 6, 7),
                            threshold = _thresholdsHE,
                            detectorEnum = cms.int32(2)
                            )
                        )
                  ),
                  cms.PSet(
                      name = cms.string("PFRecHitQTestHCALChannel"),
                      maxSeverities      = cms.vint32(11),
                      cleaningThresholds = cms.vdouble(0.0),
                      flags              = cms.vstring('Standard')
                  )
             )
           ),
    )
)

_module_pset_cuda = _module_pset.clone()
_module_pset_cuda.producers[0].src= "hbheRecHitProducerGPU" # use GPU version as input instead of legacy version
_module_pset_cuda.produceSoA = cms.bool(True)
_module_pset_cuda.produceLegacy = cms.bool(True)
_module_pset_cuda.produceCleanedLegacy = cms.bool(True)

for idx, x in enumerate(_module_pset_cuda.producers):
    for idy, y in enumerate(x.qualityTests):
        if y.name._value == "PFRecHitQTestHCALThresholdVsDepth": # when applying phase1 depth-dependent HCAL thresholds
            for idz, z in enumerate(y.cuts): # convert signed to unsigned
                if z.detectorEnum == 1: # HB
                    z.detectorEnum = cms.uint32( 1 )
                    z.depth = cms.vuint32( 1, 2, 3, 4 )
                if z.detectorEnum == 2: # HE
                    z.detectorEnum = cms.uint32( 2 )
                    z.depth = cms.vuint32( 1, 2, 3, 4, 5, 6, 7  )

_particleFlowRecHitHBHE_cpu = cms.EDProducer("PFRecHitProducer", _module_pset.clone() )
_particleFlowRecHitHBHE_cpu.produceDummyProducts=cms.bool(True)
_particleFlowRecHitHBHE_cpu.PFRecHitsGPUOut=cms.string("") 

_particleFlowRecHitHBHE_cuda = cms.EDProducer("PFHBHERecHitProducerGPU", _module_pset_cuda.clone() )

# offline 2018 -- uncollapsed
from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
from Configuration.ProcessModifiers.run2_HECollapse_2018_cff import run2_HECollapse_2018
(run2_HE_2018 & ~run2_HECollapse_2018).toModify(_particleFlowRecHitHBHE_cpu,
    producers = {0 : dict(qualityTests = {0 : dict(cuts = {1 : dict(threshold = _thresholdsHEphase1) } ) } ) },
)
(run2_HE_2018 & ~run2_HECollapse_2018).toModify(_particleFlowRecHitHBHE_cuda,
    producers = {0 : dict(qualityTests = {0 : dict(cuts = {1 : dict(threshold = _thresholdsHEphase1) } ) } ) },
)

# offline 2021
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(_particleFlowRecHitHBHE_cpu,
    producers = {0 : dict(qualityTests = {0 : dict(cuts = {0 : dict(threshold = _thresholdsHBphase1) } ) } ) },
)
run3_HB.toModify(_particleFlowRecHitHBHE_cuda,
    producers = {0 : dict(qualityTests = {0 : dict(cuts = {0 : dict(threshold = _thresholdsHBphase1) } ) } ) },
)

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
particleFlowRecHitHBHE = SwitchProducerCUDA(
    cpu = _particleFlowRecHitHBHE_cpu.clone()
)

from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toModify(particleFlowRecHitHBHE, 
    cuda = _particleFlowRecHitHBHE_cuda.clone()
)


# HCALonly WF
particleFlowRecHitHBHEOnly = _particleFlowRecHitHBHE_cpu.clone(
    producers = { 0: dict(src = "hbheprereco") }
)
run3_HB.toModify(particleFlowRecHitHBHEOnly,
    producers = { 0: dict(src = "hbhereco") }
)

# KenH: this overwrites the original above. I am not sure if that's a good thing.
particleFlowRecHitHBHEOnly = SwitchProducerCUDA(
    cpu = _particleFlowRecHitHBHE_cpu.clone()
)

from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toModify(particleFlowRecHitHBHEOnly,
    cuda = _particleFlowRecHitHBHE_cuda.clone()
)
