import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

#from RecoLocalCalo.HcalRecAlgos.HcalSeverityLevel_cfi import hcal_sev_producer

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

#process.source = cms.Source("PoolSource",
#    # replace 'myfile.root' with the source file you want to use
#    fileNames = cms.untracked.vstring(
#        'file:myfile.root'
#    )
#)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

#process.hcal_sev_producer = cms.ESProducer("HcalSeverityLevelProducer",
#     Levels = cms.VPSet(
#        cms.PSet( Level = cms.int32(0),
#                  RecHitFlags = cms.vstring(''),
#                  ChannelStatus = cms.vstring('')
#                ),
#        cms.PSet( Level = cms.int32(10),
#                  RecHitFlags = cms.vstring(''),
#                  ChannelStatus = cms.vstring('HcalCellHot')
#                ),
#        cms.PSet( Level = cms.int32(20),
#                  RecHitFlags = cms.vstring(''),
#                  ChannelStatus = cms.vstring('HcalCellOff', 'HcalCellDead')
#                )
#        )
#)

process.demo = cms.EDAnalyzer('HcalSevLvlAnalyzer',
     Levels = cms.VPSet(
        cms.PSet( Level = cms.int32(0),
                  RecHitFlags = cms.vstring(''),
                  ChannelStatus = cms.vstring('')
                ),
        cms.PSet( Level = cms.int32(10),
                  RecHitFlags = cms.vstring(''),
                  ChannelStatus = cms.vstring('HcalCellHot')
                ),
        cms.PSet( Level = cms.int32(20),
                  RecHitFlags = cms.vstring('HBHEHpdHitMultiplicity', 'HBHEPulseShape', 'HOBit',
                                            'HFDigiTime', 'HFLongShort', 'ZDCBit', 'CalibrationBit'),
                  ChannelStatus = cms.vstring('HcalCellOff', 'HcalCellDead')
                )
        )
)


process.p = cms.Path(process.demo)
