import FWCore.ParameterSet.Config as cms
from DQMOffline.Trigger.HTMonitoring_Client_cff import *
from DQMOffline.Trigger.METMonitoring_Client_cff import *

b2gjetEfficiency = cms.EDProducer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/B2GMonitor/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages                                                                                    
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_pfjetpT          'Jet pT turnON;            PFJet(pT) [GeV]; efficiency'     pfjetpT_numerator          pfjetpT_denominator",
        "effic_pfjetpT_variable 'Jet pT turnON;            PFJet(pT) [GeV]; efficiency'     pfjetpT_variable_numerator pfjetpT_variable_denominator",
        "effic_pfjetPhi         'Jet efficiency vs #phi; PF Jet #phi [rad]; efficiency'     pfjetPhi_numerator         pfjetPhi_denominator",
        "effic_pfjetEta         'Jet efficiency vs #eta; PF Jet #eta [rad]; efficiency'     pfjetEta_numerator         pfjetEta_denominator"
    ),
)

b2gClient = cms.Sequence(
    b2gjetEfficiency *
    htClient *
    metClient
)
