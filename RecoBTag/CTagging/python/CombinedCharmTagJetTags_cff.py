import FWCore.ParameterSet.Config as cms

from RecoBTag.CTagging.pfNegativeCombinedCvsLJetTags_cfi import *
from RecoBTag.CTagging.pfPositiveCombinedCvsLJetTags_cfi import *

pfNegativeCombinedCvsBJetTags = pfNegativeCombinedCvsLJetTags.clone(
   jetTagComputer = 'charmTagsNegativeComputerCvsB'
   )

pfPositiveCombinedCvsBJetTags = pfPositiveCombinedCvsLJetTags.clone(
   jetTagComputer = 'charmTagsPositiveComputerCvsB'
   )

CombinedCharmTagJetTagsTask = cms.Task(
   pfNegativeCombinedCvsLJetTags,
   pfPositiveCombinedCvsLJetTags,
   pfNegativeCombinedCvsBJetTags,
   pfPositiveCombinedCvsBJetTags
   )
