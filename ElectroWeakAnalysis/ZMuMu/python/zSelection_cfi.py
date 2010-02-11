import FWCore.ParameterSet.Config as cms

zSelection = cms.PSet(
    cut = cms.string("charge = 0 & daughter(0).pt > 10 & daughter(1).pt > 10 & abs(daughter(0).eta)<2.1 & abs(daughter(1).eta)<2.1 & mass > 0"),
    isoCut = cms.double(3.),
    ptThreshold = cms.untracked.double("1.5"),
    etEcalThreshold = cms.untracked.double("0.2"),
    etHcalThreshold = cms.untracked.double("0.5"),
    deltaRVetoTrk = cms.untracked.double("0.015"),
    deltaRTrk = cms.untracked.double("0.3"),
    deltaREcal = cms.untracked.double("0.25"),
    deltaRHcal = cms.untracked.double("0.25"),
    alpha = cms.untracked.double("0."),
    beta = cms.untracked.double("-0.75"),
    relativeIsolation = cms.bool(False)

# For standard isolation (I_Tkr<3GeV) choose this configuration:
#   isoCut = cms.double(3.),
#   ptThreshold = cms.untracked.double("1.5"),
#   etEcalThreshold = cms.untracked.double("0.2"),
#   etHcalThreshold = cms.untracked.double("0.5"),
#   deltaRVetoTrk = cms.untracked.double("0.015"),
#   deltaRTrk = cms.untracked.double("0.3"),
#   deltaREcal = cms.untracked.double("0.25"),
#   deltaRHcal = cms.untracked.double("0.25"),
#   alpha = cms.untracked.double("0."),
#   beta = cms.untracked.double("-0.75"),
#   relativeIsolation = cms.bool(False)
 )
