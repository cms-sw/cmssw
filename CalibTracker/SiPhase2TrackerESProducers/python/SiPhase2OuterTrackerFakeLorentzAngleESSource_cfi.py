import FWCore.ParameterSet.Config as cms

from CalibTracker.SiPhase2TrackerESProducers.siPhase2OTFakeLorentzAngleESSource_cfi import siPhase2OTFakeLorentzAngleESSource
SiPhase2OTFakeLorentzAngleESSource = siPhase2OTFakeLorentzAngleESSource.clone(LAValue = 0.07,
                                                                              recordName = 'LorentzAngle')
