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
# RecoveredRecHitBits: this is a mask for the determination of whether a particular RecHit is recovered
#                      empty mask means that no flag is assigned to the recovered status
#
# DropChannelStatus: this is a mask for the determination of whether a digi should be/is dropped
#                    during reconstruction because of the channelstatus of its cell
#                    empty mask means that no digi should be/is dropped
#
# Modified 21.09.2010: a level consisting only of invalid definitions will be ignored
# Any errors in definition of severity levels come through LogWarning

essourceSev =  cms.ESSource("EmptyESSource",
                   recordName = cms.string("HcalSeverityLevelComputerRcd"),
                   firstValid = cms.vuint32(1),
                   iovIsRunNotTime = cms.bool(True)
)


hcalRecAlgos = cms.ESProducer("HcalRecAlgoESProducer",
    SeverityLevels = cms.VPSet(
        # the following is the default level, please do not modify its definition:
        cms.PSet( Level = cms.int32(0),
                  RecHitFlags = cms.vstring(''),
                  ChannelStatus = cms.vstring('')
                ),
        cms.PSet( Level = cms.int32(1),
                  RecHitFlags = cms.vstring(''),
                  ChannelStatus = cms.vstring('HcalCellCaloTowerProb')
                ),
        cms.PSet( Level = cms.int32(5),
                  RecHitFlags = cms.vstring('HSCP_R1R2','HSCP_FracLeader','HSCP_OuterEnergy',
                                            'HSCP_ExpFit','ADCSaturationBit', 'HBHEIsolatedNoise',
                                            'AddedSimHcalNoise'),
                  ChannelStatus = cms.vstring('HcalCellExcludeFromHBHENoiseSummary')
                ),
        cms.PSet( Level = cms.int32(8),
                  RecHitFlags = cms.vstring('HBHEHpdHitMultiplicity', 
                                            'HBHEPulseShape', 
                                            'HOBit',
                                            'HFDigiTime',
                                            'HFInTimeWindow',
                                            'ZDCBit', 'CalibrationBit',
                                            'TimingErrorBit',
                                            'HBHEFlatNoise',
                                            'HBHESpikeNoise',
                                            'HBHETriangleNoise',
                                            'HBHETS4TS5Noise'
                                           ),
                  ChannelStatus = cms.vstring('')
                ),
        # March 2010:  HFLongShort now tested, and should be excluded from CaloTowers by default
        cms.PSet( Level = cms.int32(11),
                  RecHitFlags = cms.vstring('HFLongShort',
                                            # HFPET and HFS8S1Ratio feed HFLongShort, and should be at the same severity
                                            'HFPET',
                                            'HFS8S1Ratio'
                                            #'HFDigiTime'  # This should be set to 11 in data ONLY.  We can't set it to 11 by default, because default values should reflect MC settings, and the flag can't be used in MC
                                            ),
                  ChannelStatus = cms.vstring('')
                ),
        cms.PSet( Level = cms.int32(12),
                  RecHitFlags = cms.vstring(''),
                  ChannelStatus = cms.vstring('HcalCellCaloTowerMask')
                ),
        cms.PSet( Level = cms.int32(15),
                  RecHitFlags = cms.vstring(''),
                  ChannelStatus = cms.vstring('HcalCellHot')
                ),
        cms.PSet( Level = cms.int32(20),
                  RecHitFlags = cms.vstring(''),
                  ChannelStatus = cms.vstring('HcalCellOff', 'HcalCellDead')
                )
        ),
    RecoveredRecHitBits = cms.vstring('TimingAddedBit','TimingSubtractedBit'),
    DropChannelStatusBits = cms.vstring('HcalCellMask','HcalCellOff', 'HcalCellDead')
)
