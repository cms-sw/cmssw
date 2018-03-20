import FWCore.ParameterSet.Config as cms
import RecoLocalCalo.HcalRecProducers.HBHEMethod2Parameters_cfi as method2
import RecoLocalCalo.HcalRecProducers.HBHEMahiParameters_cfi as mahi

mahiDebugger = cms.EDAnalyzer('MahiDebugger',
                              cms.PSet( applyTimeSlew = method2.m2Parameters.applyTimeSlew,
                                        meanTime = method2.m2Parameters.meanTime,
                                        timeSigmaHPD = method2.m2Parameters.timeSigmaHPD,
                                        timeSigmaSiPM = method2.m2Parameters.timeSigmaSiPM),
                              mahi.mahiParameters)
