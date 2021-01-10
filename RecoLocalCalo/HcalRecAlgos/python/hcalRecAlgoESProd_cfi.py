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

from RecoLocalCalo.HcalRecAlgos.hcalRecAlgos_cfi import hcalRecAlgos

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify(hcalRecAlgos,
    phase = 1,
    SeverityLevels = {
        2 : dict( RecHitFlags = ['HBHEIsolatedNoise',
                                 'HFAnomalousHit']
            ),
        3 : dict( RecHitFlags = ['HBHEHpdHitMultiplicity',  
                                 'HBHEFlatNoise', 
                                 'HBHESpikeNoise', 
                                 'HBHETS4TS5Noise', 
                                 'HBHENegativeNoise', 
                                 'HBHEOOTPU']
            ),
        4 : dict( RecHitFlags = ['HFLongShort', 
                                 'HFS8S1Ratio',  
                                 'HFPET', 
                                 'HFSignalAsymmetry']
            ),
    },
    RecoveredRecHitBits = ['']
)
