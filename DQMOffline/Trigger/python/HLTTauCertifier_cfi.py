import FWCore.ParameterSet.Config as cms

HLTTauCertification = cms.EDFilter("HLTTauCertifier",
                                   targetDir = cms.string("EventInfo"),
                                   targetME  = cms.string("HLT_Tau"),
                                   inputMEs_ = cms.vstring(
                                      "HLT/TauOffline/Photons/DoubleTau/EfficiencyRefPrevious",
                                      "HLT/TauOffline/Photons/SingleTau/EfficiencyRefPrevious"
                                   ),
                                   setBadRunOnWarnings = cms.bool(False),
                                   setBadRunOnErrors   = cms.bool(True),
)



