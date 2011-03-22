import FWCore.ParameterSet.Config as cms

import DQM.HLTEvF.hltMonBTagIPSource_cfi
import DQM.HLTEvF.hltMonBTagMuSource_cfi

# definition of the Sources for 8E29
#hltMonBTagIP_Jet50U_Source = DQM.HLTEvF.hltMonBTagIPSource_cfi.hltMonBTagIPSource.clone()
#hltMonBTagMu_Jet10U_Source = DQM.HLTEvF.hltMonBTagMuSource_cfi.hltMonBTagMuSource.clone()

#hltMonBTag = cms.Path( hltMonBTagIP_Jet50U_Source + hltMonBTagMu_Jet10U_Source )


# simple b-tag monitor (can also just be included through HLTMonSimpleBTag_cff)
import DQM.HLTEvF.HLTMonSimpleBTag_cfi
hltMonBTag = cms.Path(hltMonSimpleBTag)
