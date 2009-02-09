import FWCore.ParameterSet.Config as cms

# HcalSeverityLevelComputer:
# short instruction: severity levels are defined to grade a problem indicated through flags
#
#  The algorithm works from the highest level down. For each level, it determines whether any of the
#  bits that is defined for its level is set. If yes, then - regardless of the setting of the other bits -
#  it gives back the corresponding severity level, if no, it continues with the next lower level.
#  If a defined bit vector is empty, the corresponding flag is not checked.
#  This means that the highest level that has two empty vectors will be always the default level.
# 

essourceSev =  cms.ESSource("EmptyESSource",
                   recordName = cms.string("HcalSeverityLevelComputerRcd"),
                   firstValid = cms.vuint32(1),
                   iovIsRunNotTime = cms.bool(True)
)


hcalRecAlgos = cms.ESProducer("HcalRecAlgoESProducer",
    SeverityLevels = cms.VPSet(
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
