import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *

#-------------------------------------------------
# Include masking only from Cabling and O2O
#-------------------------------------------------

siStripQualityESProducer.appendToDataLabel = 'unbiased'
siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
    cms.PSet(
        record = cms.string( 'SiStripDetCablingRcd' ), # bad components from cabling
        tag = cms.string( '' )
    ),
    cms.PSet(
        record = cms.string( 'SiStripBadChannelRcd' ), # bad components from O2O
        tag = cms.string( '' )
    )
)

#import CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi
#siStripQualityESProducerUnbiased = CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi.siStripQualityESProducer.clone()
#siStripQualityESProducerUnbiased.appendToDataLabel = 'unbiased'
#siStripQualityESProducerUnbiased.ListOfRecordToMerge = cms.VPSet(
#    cms.PSet(
#        record = cms.string( 'SiStripDetCablingRcd' ), # bad components from cabling
#        tag = cms.string( '' )
#    ),
#    cms.PSet(
#        record = cms.string( 'SiStripBadChannelRcd' ), # bad components from O2O
#        tag = cms.string( '' )
#    )
#)
