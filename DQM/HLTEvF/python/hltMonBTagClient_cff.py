import FWCore.ParameterSet.Config as cms

import DQM.HLTEvF.hltMonBTagIPClient_cfi
import DQM.HLTEvF.hltMonBTagMuClient_cfi

# definition of the Clients for 8E29
hltMonBTagIP_Jet50U_Client = DQM.HLTEvF.hltMonBTagIPClient_cfi.hltMonBTagIPClient.clone()
hltMonBTagIP_Jet50U_Client.updateLuminosityBlock = True

hltMonBTagMu_Jet10U_Client = DQM.HLTEvF.hltMonBTagMuClient_cfi.hltMonBTagMuClient.clone()
hltMonBTagMu_Jet10U_Client.updateLuminosityBlock = True

hltMonBTagClient = cms.Sequence( hltMonBTagIP_Jet50U_Client + hltMonBTagMu_Jet10U_Client )
