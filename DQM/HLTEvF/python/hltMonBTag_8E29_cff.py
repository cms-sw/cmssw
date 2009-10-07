import FWCore.ParameterSet.Config as cms

import DQM.HLTEvF.hltMonBTagIPSource_cfi
import DQM.HLTEvF.hltMonBTagMuSource_cfi
import DQM.HLTEvF.hltMonBTagIPClient_cfi
import DQM.HLTEvF.hltMonBTagMuClient_cfi

# definition of the Sources for 8E29
hltMonBTagIP_Jet50U_Source = DQM.HLTEvF.hltMonBTagIPSource_cfi.hltMonBTagIPSource.clone()
hltMonBTagMu_Jet10U_Source = DQM.HLTEvF.hltMonBTagMuSource_cfi.hltMonBTagMuSource.clone()

hltMonBTagSource_8E29 = cms.Sequence( hltMonBTagIP_Jet50U_Source + hltMonBTagMu_Jet10U_Source )

# definition of the Clients for 8E29
hltMonBTagIP_Jet50U_Client = DQM.HLTEvF.hltMonBTagIPClient_cfi.hltMonBTagIPClient.clone()
hltMonBTagMu_Jet10U_Client = DQM.HLTEvF.hltMonBTagMuClient_cfi.hltMonBTagMuClient.clone()

hltMonBTagClient_8E29 = cms.Sequence( hltMonBTagIP_Jet50U_Client + hltMonBTagMu_Jet10U_Client )
