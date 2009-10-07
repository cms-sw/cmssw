import FWCore.ParameterSet.Config as cms

import DQM.HLTEvF.hltMonBTagIPSource_cfi
import DQM.HLTEvF.hltMonBTagMuSource_cfi
import DQM.HLTEvF.hltMonBTagIPClient_cfi
import DQM.HLTEvF.hltMonBTagMuClient_cfi

# definition of the Sources for 1E31
hltMonBTagIP_Jet80_Source = DQM.HLTEvF.hltMonBTagIPSource_cfi.hltMonBTagIPSource.clone()
hltMonBTagIP_Jet80_Source.pathName    = 'HLT_BTagIP_Jet80'
hltMonBTagIP_Jet80_Source.L1Filter    = 'hltL1sBTagIPJet80'
hltMonBTagIP_Jet80_Source.L2Filter    = 'hltBJet80'
hltMonBTagIP_Jet80_Source.L2Jets      = 'hltMCJetCorJetIcone5Regional'
hltMonBTagIP_Jet80_Source.L25TagInfo  = 'hltBLifetimeL25TagInfosStartup'
hltMonBTagIP_Jet80_Source.L25JetTags  = 'hltBLifetimeL25BJetTagsStartup'
hltMonBTagIP_Jet80_Source.L25Filter   = 'hltBLifetimeL25FilterStartup'
hltMonBTagIP_Jet80_Source.L3TagInfo   = 'hltBLifetimeL3TagInfosStartup'
hltMonBTagIP_Jet80_Source.L3JetTags   = 'hltBLifetimeL3BJetTagsStartup'
hltMonBTagIP_Jet80_Source.L3Filter    = 'hltBLifetimeL3FilterStartup'

hltMonBTagIP_Jet120_Source = DQM.HLTEvF.hltMonBTagIPSource_cfi.hltMonBTagIPSource.clone()
hltMonBTagIP_Jet120_Source.pathName   = 'HLT_BTagIP_Jet120'
hltMonBTagIP_Jet120_Source.L1Filter   = 'hltL1sBTagIPJet120'
hltMonBTagIP_Jet120_Source.L2Filter   = 'hltBJet120'
hltMonBTagIP_Jet120_Source.L2Jets     = 'hltMCJetCorJetIcone5Regional'
hltMonBTagIP_Jet120_Source.L25TagInfo = 'hltBLifetimeL25TagInfosStartup'
hltMonBTagIP_Jet120_Source.L25JetTags = 'hltBLifetimeL25BJetTagsStartup'
hltMonBTagIP_Jet120_Source.L25Filter  = 'hltBLifetimeL25FilterStartup'
hltMonBTagIP_Jet120_Source.L3TagInfo  = 'hltBLifetimeL3TagInfosStartup'
hltMonBTagIP_Jet120_Source.L3JetTags  = 'hltBLifetimeL3BJetTagsStartup'
hltMonBTagIP_Jet120_Source.L3Filter   = 'hltBLifetimeL3FilterStartup'

hltMonBTagMu_Jet20_Source = DQM.HLTEvF.hltMonBTagMuSource_cfi.hltMonBTagMuSource.clone()
hltMonBTagMu_Jet20_Source.pathName    = 'HLT_BTagMu_Jet20'
hltMonBTagMu_Jet20_Source.L1Filter    = 'hltL1sBTagMuJet20'
hltMonBTagMu_Jet20_Source.L2Filter    = 'hltBJet20'
hltMonBTagMu_Jet20_Source.L2Jets      = 'hltMCJetCorJetIcone5'
hltMonBTagMu_Jet20_Source.L25TagInfo  = 'hltBSoftMuonL25TagInfos'
hltMonBTagMu_Jet20_Source.L25JetTags  = 'hltBSoftMuonL25BJetTagsByDR'
hltMonBTagMu_Jet20_Source.L25Filter   = 'hltBSoftMuonL25FilterByDR'
hltMonBTagMu_Jet20_Source.L3TagInfo   = 'hltBSoftMuonL3TagInfos'
hltMonBTagMu_Jet20_Source.L3JetTags   = 'hltBSoftMuonL3BJetTagsByDR'
hltMonBTagMu_Jet20_Source.L3Filter    = 'hltBSoftMuonL3FilterByDR'

hltMonBTagSource_1E31 = cms.Sequence( hltMonBTagIP_Jet80_Source + hltMonBTagMu_Jet20_Source + hltMonBTagIP_Jet120_Source )

# definition of the Clients for 1E31
hltMonBTagIP_Jet80_Client = DQM.HLTEvF.hltMonBTagIPClient_cfi.hltMonBTagIPClient.clone()
hltMonBTagIP_Jet80_Client.pathName    = 'HLT_BTagIP_Jet80'

hltMonBTagIP_Jet120_Client = DQM.HLTEvF.hltMonBTagIPClient_cfi.hltMonBTagIPClient.clone()
hltMonBTagIP_Jet120_Client.pathName   = 'HLT_BTagIP_Jet120'

hltMonBTagMu_Jet20_Client = DQM.HLTEvF.hltMonBTagMuClient_cfi.hltMonBTagMuClient.clone()
hltMonBTagMu_Jet20_Client.pathName    = 'HLT_BTagMu_Jet20'

hltMonBTagClient_1E31 = cms.Sequence( hltMonBTagIP_Jet80_Client + hltMonBTagMu_Jet20_Client + hltMonBTagIP_Jet120_Client )
