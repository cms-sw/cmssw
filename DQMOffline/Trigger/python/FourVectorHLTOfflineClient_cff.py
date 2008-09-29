import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.FourVectorHLTOfflineClient_cfi import *
hltFourVectorSeqClient = cms.Sequence(hltFourVectorClient)

