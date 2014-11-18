import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoBTag.ImpactParameter.impactParameterTagInfos_cfi import *
from RecoBTag.ImpactParameter.pfImpactParameterTagInfos_cfi import *
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
from RecoBTag.ImpactParameter.negativeOnlyJetBProbabilityBJetTags_cfi import *
# Jet Prob
from RecoBTag.ImpactParameter.negativeOnlyJetProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.negativeOnlyJetProbabilityBJetTags_cfi import *
# High Eff
from RecoBTag.ImpactParameter.negativeTrackCounting3D2ndComputer_cfi import *
from RecoBTag.ImpactParameter.negativeTrackCountingHighEffBJetTags_cfi import *
# High Purity
from RecoBTag.ImpactParameter.negativeTrackCounting3D3rdComputer_cfi import *
from RecoBTag.ImpactParameter.negativeTrackCountingHighPurBJetTags_cfi import *

# Positive-only Tags

# Jet BProb
from RecoBTag.ImpactParameter.positiveOnlyJetBProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.positiveOnlyJetBProbabilityBJetTags_cfi import *
# Jet Prob
from RecoBTag.ImpactParameter.positiveOnlyJetProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.positiveOnlyJetProbabilityBJetTags_cfi import *


# New candidate based fwk

# Jet BProb
from RecoBTag.ImpactParameter.candidateJetBProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.pfJetBProbabilityBJetTags_cfi import *
# Jet Prob
from RecoBTag.ImpactParameter.candidateJetProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.pfJetProbabilityBJetTags_cfi import *
# High Eff
from RecoBTag.ImpactParameter.candidateTrackCounting3D2ndComputer_cfi import *
from RecoBTag.ImpactParameter.pfTrackCountingHighEffBJetTags_cfi import *
# High Purity
from RecoBTag.ImpactParameter.candidateTrackCounting3D3rdComputer_cfi import *
from RecoBTag.ImpactParameter.pfTrackCountingHighPurBJetTags_cfi import *


