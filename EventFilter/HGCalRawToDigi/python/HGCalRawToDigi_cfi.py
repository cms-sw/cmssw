import FWCore.ParameterSet.Config as cms
import EventFilter.HGCalRawToDigi.HGCalRawToDigiFake_cfi

hgcalDigis = EventFilter.HGCalRawToDigi.HGCalRawToDigiFake_cfi.HGCalRawToDigiFake.clone()

import EventFilter.HGCalRawToDigi.HFNoseRawToDigiFake_cfi

hfnoseDigis = EventFilter.HGCalRawToDigi.HFNoseRawToDigiFake_cfi.HFNoseRawToDigiFake.clone()

from Configuration.ProcessModifiers.convertHGCalDigisReco_cff import convertHGCalDigisReco
import EventFilter.HGCalRawToDigi.HGCDigiConverter_cfi
_hgcalDigisConverted = EventFilter.HGCalRawToDigi.HGCDigiConverter_cfi.HGCDigiConverter.clone()
convertHGCalDigisReco.toReplaceWith(hgcalDigis,_hgcalDigisConverted)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hgcalDigis,
    eeDigis = 'mixData:HGCDigisEE',
    fhDigis = 'mixData:HGCDigisHEfront',
    bhDigis = 'mixData:HGCDigisHEback',
)

