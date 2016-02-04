import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoBTag.ImpactParameter.impactParameter_cfi import *
#tmp fix, use fake/frontier
#include "RecoBTag/Configuration/data/RecoBTag_FakeConditions.cff"
from RecoBTau.JetTagComputer.jetTagRecord_cfi import *
# MVA
from RecoBTag.ImpactParameter.impactParameterMVAComputer_cfi import *
from RecoBTag.ImpactParameter.impactParameterMVABJetTags_cfi import *
#B Jet Prob
from RecoBTag.ImpactParameter.jetBProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.jetBProbabilityBJetTags_cfi import *
#Jet Prob
from RecoBTag.ImpactParameter.jetProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.jetProbabilityBJetTags_cfi import *
# HighEff
from RecoBTag.ImpactParameter.trackCounting3D2ndComputer_cfi import *
from RecoBTag.ImpactParameter.trackCountingHighEffBJetTags_cfi import *
# High Purity
from RecoBTag.ImpactParameter.trackCounting3D3rdComputer_cfi import *
from RecoBTag.ImpactParameter.trackCountingHighPurBJetTags_cfi import *

