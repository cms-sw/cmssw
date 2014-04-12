import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoBTag.ImpactParameter.impactParameter_cfi import *
#tmp fix, use fake/frontier
#include "RecoBTag/Configuration/data/RecoBTag_FakeConditions.cff"
from RecoBTau.JetTagComputer.jetTagRecord_cfi import *
# MVA
from RecoBTag.ImpactParameter.impactParameterMVAComputer_cfi import *
from RecoBTag.ImpactParameter.impactParameterMVABJetTags_cfi import *
# Jet BProb
from RecoBTag.ImpactParameter.jetBProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.jetBProbabilityBJetTags_cfi import *
# Jet Prob
from RecoBTag.ImpactParameter.jetProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.jetProbabilityBJetTags_cfi import *
# High Eff
from RecoBTag.ImpactParameter.trackCounting3D2ndComputer_cfi import *
from RecoBTag.ImpactParameter.trackCountingHighEffBJetTags_cfi import *
# High Purity
from RecoBTag.ImpactParameter.trackCounting3D3rdComputer_cfi import *
from RecoBTag.ImpactParameter.trackCountingHighPurBJetTags_cfi import *

# Negative Tags

# Jet BProb
from RecoBTag.ImpactParameter.negativeOnlyJetBProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.negativeOnlyJetBProbabilityJetTags_cfi import *
# Jet Prob
from RecoBTag.ImpactParameter.negativeOnlyJetProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.negativeOnlyJetProbabilityJetTags_cfi import *
# High Eff
from RecoBTag.ImpactParameter.negativeTrackCounting3D2ndComputer_cfi import *
from RecoBTag.ImpactParameter.negativeTrackCountingHighEffJetTags_cfi import *
# High Purity
from RecoBTag.ImpactParameter.negativeTrackCounting3D3rdComputer_cfi import *
from RecoBTag.ImpactParameter.negativeTrackCountingHighPurJetTags_cfi import *

# Positive Tags

# Jet BProb
from RecoBTag.ImpactParameter.positiveOnlyJetBProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.positiveOnlyJetBProbabilityJetTags_cfi import *
# Jet Prob
from RecoBTag.ImpactParameter.positiveOnlyJetProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.positiveOnlyJetProbabilityJetTags_cfi import *


