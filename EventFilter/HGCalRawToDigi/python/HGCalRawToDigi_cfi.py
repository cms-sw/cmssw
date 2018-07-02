import FWCore.ParameterSet.Config as cms
import EventFilter.HGCalRawToDigi.HGCalRawToDigiFake_cfi

hgcalDigis = EventFilter.HGCalRawToDigi.HGCalRawToDigiFake_cfi.HGCalRawToDigiFake.clone()

from Configuration.ProcessModifiers.convertHGCalDigis_cff import convertHGCalDigis
import EventFilter.HGCalRawToDigi.HGCalDigiConverter_cfi
_hgcalDigisConverted = EventFilter.HGCalRawToDigi.HGCalDigiConverter_cfi.HGCalDigiConverter.clone()
convertHGCalDigis.toReplaceWith(hgcalDigis,_hgcalDigisConverted)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(HGCalUncalibRecHit,
    eeDigis = 'mixData:HGCDigisEE',
    fhDigis = 'mixData:HGCDigisHEfront',
    bhDigis = 'mixData:HGCDigisHEback',
)

