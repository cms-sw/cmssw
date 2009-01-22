import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )


process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

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
# short instruction:
#  severity levels are defined to grade a problem indicated through flags
#
#  The algorithm works from the highest level down. For each level, it determines whether any of the
#  bits that is defined for its level is set. If yes, then - regardless of the setting of the other bits -
#  it gives back the corresponding severity level, if no, it continues with the next lower level.
#  If a defined bit vector is empty, the corresponding flag is not checked.
#  This means that the highest level that has two empty vectors will be always the default level.
# 
)


process.p = cms.Path(process.demo)
