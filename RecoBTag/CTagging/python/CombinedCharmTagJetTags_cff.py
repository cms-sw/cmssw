import FWCore.ParameterSet.Config as cms

from RecoBTag.CTagging.pfNegativeCombinedCvsLJetTags_cfi import *
from RecoBTag.CTagging.pfPositiveCombinedCvsLJetTags_cfi import *

pfNegativeCombinedCvsBJetTags = pfNegativeCombinedCvsLJetTags.clone(
   jetTagComputer = cms.string('charmTagsNegativeComputerCvsB')
   )

pfPositiveCombinedCvsBJetTags = pfPositiveCombinedCvsLJetTags.clone(
   jetTagComputer = cms.string('charmTagsPositiveComputerCvsB')
   )
