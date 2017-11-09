import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.JetMonitor_cff import *

jetmetMonitorHLT = cms.Sequence(
    HLTJetmonitoring
)

jmeHLTDQMSourceExtra = cms.Sequence(
    PFJet40_Prommonitoring    
    *PFJet60_Prommonitoring    
    *PFJet80_Prommonitoring    
    *PFJet140_Prommonitoring    
    *PFJet200_Prommonitoring    
    *PFJet260_Prommonitoring    
    *PFJet320_Prommonitoring    
    *PFJet400_Prommonitoring
    *PFJet450_Prommonitoring
    *PFJet500_Prommonitoring
    *PFJetFwd40_Prommonitoring    
    *PFJetFwd60_Prommonitoring    
    *PFJetFwd80_Prommonitoring    
    *PFJetFwd140_Prommonitoring    
    *PFJetFwd200_Prommonitoring    
    *PFJetFwd260_Prommonitoring    
    *PFJetFwd320_Prommonitoring    
    *PFJetFwd400_Prommonitoring    
    *PFJetFwd450_Prommonitoring
    *PFJetFwd500_Prommonitoring
    *AK8PFJet450_Prommonitoring
    *AK8PFJet40_Prommonitoring    
    *AK8PFJet60_Prommonitoring    
    *AK8PFJet80_Prommonitoring    
    *AK8PFJet140_Prommonitoring    
    *AK8PFJet200_Prommonitoring    
    *AK8PFJet260_Prommonitoring    
    *AK8PFJet320_Prommonitoring    
    *AK8PFJet400_Prommonitoring    
    *AK8PFJet500_Prommonitoring
    *AK8PFJetFwd450_Prommonitoring
    *AK8PFJetFwd40_Prommonitoring    
    *AK8PFJetFwd60_Prommonitoring    
    *AK8PFJetFwd80_Prommonitoring    
    *AK8PFJetFwd140_Prommonitoring    
    *AK8PFJetFwd200_Prommonitoring    
    *AK8PFJetFwd260_Prommonitoring    
    *AK8PFJetFwd320_Prommonitoring    
    *AK8PFJetFwd400_Prommonitoring    
    *AK8PFJetFwd500_Prommonitoring 
    *CaloJet500_NoJetID_Prommonitoring 
)
