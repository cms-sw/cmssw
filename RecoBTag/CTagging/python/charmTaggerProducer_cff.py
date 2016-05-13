import FWCore.ParameterSet.Config as cms

from RecoBTag.CTagging.charmTagsComputerCvsL_cfi import *

charmTagsComputerCvsB = charmTagsComputerCvsL.clone(
   weightFile = cms.FileInPath('RecoBTag/CTagging/data/c_vs_b_sklearn.weight.xml'),   
   variables = c_vs_b_vars_vpset
   )

#
# Negative tagger
#

charmTagsNegativeComputerCvsL = charmTagsComputerCvsL.clone()

charmTagsNegativeComputerCvsL.slComputerCfg.vertexFlip = cms.bool(True)
charmTagsNegativeComputerCvsL.slComputerCfg.trackFlip = cms.bool(True)
charmTagsNegativeComputerCvsL.slComputerCfg.trackSelection.sip3dSigMax = 0
charmTagsNegativeComputerCvsL.slComputerCfg.trackPseudoSelection.sip3dSigMax = 0
charmTagsNegativeComputerCvsL.slComputerCfg.trackPseudoSelection.sip2dSigMin = -99999.9
charmTagsNegativeComputerCvsL.slComputerCfg.trackPseudoSelection.sip2dSigMax = -2.0

charmTagsNegativeComputerCvsB = charmTagsComputerCvsB.clone()

charmTagsNegativeComputerCvsB.slComputerCfg.vertexFlip = cms.bool(True)
charmTagsNegativeComputerCvsB.slComputerCfg.trackFlip = cms.bool(True)
charmTagsNegativeComputerCvsB.slComputerCfg.trackSelection.sip3dSigMax = 0
charmTagsNegativeComputerCvsB.slComputerCfg.trackPseudoSelection.sip3dSigMax = 0
charmTagsNegativeComputerCvsB.slComputerCfg.trackPseudoSelection.sip2dSigMin = -99999.9
charmTagsNegativeComputerCvsB.slComputerCfg.trackPseudoSelection.sip2dSigMax = -2.0

#
# Positive tagger
#

charmTagsPositiveComputerCvsL = charmTagsComputerCvsL.clone(
)

charmTagsPositiveComputerCvsL.slComputerCfg.trackSelection.sip3dSigMin = 0
charmTagsPositiveComputerCvsL.slComputerCfg.trackPseudoSelection.sip3dSigMin = 0

charmTagsPositiveComputerCvsB = charmTagsComputerCvsB.clone(
)

charmTagsPositiveComputerCvsB.slComputerCfg.trackSelection.sip3dSigMin = 0
charmTagsPositiveComputerCvsB.slComputerCfg.trackPseudoSelection.sip3dSigMin = 0
