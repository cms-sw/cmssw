import FWCore.ParameterSet.Config as cms


import HLTrigger.HLTfilters.hltHighLevel_cfi
noiseHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    eventSetupPathsKey='HcalCalNoise',
    throw = False #dont throw except on unknown path name
)

prescaler = cms.EDFilter("PrescalerFHN",
                         TriggerResultsTag = cms.InputTag("TriggerResults", "", "HLT"),

                         # Will select OR of all specified HLTs
                         # And increment if HLT is seen, regardless if
                         # others cause selection

                         Prescales = cms.VPSet(
                             cms.PSet(
#                                 HLTName = cms.string("HLT_PFMET130"),
                                 PrescaleFactor = cms.uint32(1)
                             )
                             #    cms.PSet(
                             #    name = cms.string("HLTPath2"),
                             #    factor = cms.uint32(100)
                             #    )
                         ))

from Calibration.HcalAlCaRecoProducers.alcahcalnoise_cfi import *

#seqALCARECOHcalCalNoise = cms.Sequence(noiseHLT*prescaler*HcalNoiseProd)
seqALCARECOHcalCalNoise = cms.Sequence(noiseHLT*HcalNoiseProd)
