import FWCore.ParameterSet.Config as cms

EBPhase2TPGParamProducer = cms.EDAnalyzer("EcalEBPhase2TPParamProducer",
inputFile = cms.FileInPath('SimCalorimetry/EcalEBTrigPrimProducers/data/CMSSWPhaseIIPulseGraphAlt.root'),
outputFile = cms.untracked.string('../../../SimCalorimetry/EcalEBTrigPrimProducers/data/AmpTimeOnPeakXtalWeightsCMSSWPulse_8samples_peakOnSix_WithAndyFixes.txt.gz'),
                                                nSamplesToUse = cms.uint32(8),
                                                useBXPlusOne = cms.bool(False),
                                                phaseShift  = cms.double (2.581),
                                                nWeightGroups = cms.uint32(61200),
                                                Et_sat = cms.double(1998.36),
                                                xtal_LSB = cms.double(0.0488),
                                                binOfMaximum = cms.uint32(6)


## allowed values of nSamplesToUse: 12, 8, 6. nSamplesToUse=8 is the default
## allowed values of binOfMaximum: 6, 8.    binOfMaximum=6 is the default 
#### The DEFAULT is nSamplesToUse=8,  binOfMaximum=6 #######
 
## If nSamplesToUse is 12 ==> useBXPlusOne is True                                                                                                                                               
## If nSamplesToUse is 8 ==> useBXPlusOne is False                                                                                                                                               
## If nSamplesToUse is 6 ==> useBXPlusOne is False                                                                                                                                               

)

