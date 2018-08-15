import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.BTVHLTOfflineSource_cfi import *
from DQMOffline.Trigger.BTaggingMonitoring_cff import *

btagMonitorHLT = cms.Sequence(
    BTagMu_AK4DiJet20_Mu5
    + BTagMu_AK4DiJet40_Mu5
    + BTagMu_AK4DiJet70_Mu5
    + BTagMu_AK4DiJet110_Mu5    
    + BTagMu_AK4DiJet170_Mu5
    + BTagMu_AK8DiJet170_Mu5
    + BTagMu_AK8Jet170_DoubleMu5
    + BTagMu_AK4Jet300_Mu5
    + BTagMu_AK8Jet300_Mu5
)

btvHLTDQMSourceExtra = cms.Sequence(
    PFJet40
    + PFJet60
    + PFJet80
    + PFJet140
    + PFJet200
    + PFJet260
    + PFJet320
    + PFJet400
    + PFJet450
    + PFJet500
    + PFJet550
    + AK8PFJet40
    + AK8PFJet60
    + AK8PFJet80
    + AK8PFJet140
    + AK8PFJet200
    + AK8PFJet260
    + AK8PFJet320
    + AK8PFJet400
    + AK8PFJet450
    + AK8PFJet500
    + AK8PFJet550
    + PFJetFwd40
    + PFJetFwd60
    + PFJetFwd80
    + PFJetFwd140
    + PFJetFwd200
    + PFJetFwd260
    + PFJetFwd320
    + PFJetFwd400
    + PFJetFwd450
    + PFJetFwd500
    + AK8PFJetFwd40
    + AK8PFJetFwd60
    + AK8PFJetFwd80
    + AK8PFJetFwd140
    + AK8PFJetFwd200
    + AK8PFJetFwd260
    + AK8PFJetFwd320
    + AK8PFJetFwd400
    + AK8PFJetFwd450
    + AK8PFJetFwd500
)
