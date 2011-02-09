import FWCore.ParameterSet.Config as cms

from Alignment.LaserAlignment.LaserAlignmentDefs_cff import *

RawDataConverter = cms.EDAnalyzer(
    "RawDataConverter",
    # list of digi producers
    DigiProducersList = cms.VPSet(
        cms.PSet(
            DigiLabel = cms.string( 'ZeroSuppressed' ),
            DigiProducer = cms.string( 'laserAlignmentT0Producer' ), #simSiStripDigis
            DigiType = cms.string( 'Processed' )
            ),
        cms.PSet(
            DigiLabel = cms.string('ZeroSuppressed'),
            DigiType = cms.string('Processed'),
            DigiProducer = cms.string('siStripDigis')
            ),
        cms.PSet(
            DigiLabel = cms.string('VirginRaw'),
            DigiType = cms.string('Raw'),
            DigiProducer = cms.string('siStripDigis')
            ), 
        cms.PSet(
            DigiLabel = cms.string('ProcessedRaw'),
            DigiType = cms.string('Raw'),
            DigiProducer = cms.string('siStripDigis')
            ), 
        cms.PSet(
            DigiLabel = cms.string('ScopeMode'),
            DigiType = cms.string('Raw'),
            DigiProducer = cms.string('siStripDigis')
            )
        ),
    DigiModuleLabels = cms.vstring(
        'laserAlignmentT0Producer',
        'siStripDigis'
        ),
    ProductInstanceLabels = cms.vstring(
        'ZeroSuppressed',
        'VirginRaw'
        ),
    DetIds = cms.VPSet()
    )

RawDataConverter.DetIds.extend(DETIDS_TECP_R4)
RawDataConverter.DetIds.extend(DETIDS_TECP_R6)
RawDataConverter.DetIds.extend(DETIDS_TECM_R4)
RawDataConverter.DetIds.extend(DETIDS_TECM_R6)
RawDataConverter.DetIds.extend(DETIDS_TIB)
RawDataConverter.DetIds.extend(DETIDS_TOB)
RawDataConverter.DetIds.extend(DETIDS_TECP_AT)
RawDataConverter.DetIds.extend(DETIDS_TECM_AT)
