import FWCore.ParameterSet.Config as cms

# LaserAlignmentHLT filter to be run at hLT on calibration stream to extract only Laser Alignment Modules
#
# VPSet DigiProducersList is a list of all
# possible input digi products as defined in:
# EventFilter/SiStripRawToDigiModule/plugins/SiStripRawToDigiModule.cc
#
# The additional DigiType has to be specified to tell the
# producer if the corresponding DetSet contains 
# SiStripDigis (="Processed") or SiStripRawDigis (="Raw").
# With this feature we keep the possibility to switch to
# another producer if necessary by changing only the .cfg
#
LaserAlignmentHLT = cms.EDProducer("LaserAlignmentT0Producer",
    DigiProducerList = cms.VPSet(cms.PSet(
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
        ))
)
