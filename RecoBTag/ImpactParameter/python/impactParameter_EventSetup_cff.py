import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoBTau.JetTagComputer.jetTagRecord_cfi import *
# MVA
from RecoBTag.ImpactParameter.impactParameterMVAComputer_cfi import *
# Jet BProb
from RecoBTag.ImpactParameter.jetBProbabilityComputer_cfi import *
# Jet Prob
from RecoBTag.ImpactParameter.jetProbabilityComputer_cfi import *
# High Eff
from RecoBTag.ImpactParameter.trackCounting3D2ndComputer_cfi import *
# High Purity
from RecoBTag.ImpactParameter.trackCounting3D3rdComputer_cfi import *

# Negative Tags

# Jet BProb
from RecoBTag.ImpactParameter.negativeOnlyJetBProbabilityComputer_cfi import *
# Jet Prob
from RecoBTag.ImpactParameter.negativeOnlyJetProbabilityComputer_cfi import *
# High Eff
from RecoBTag.ImpactParameter.negativeTrackCounting3D2ndComputer_cfi import *
# High Purity
from RecoBTag.ImpactParameter.negativeTrackCounting3D3rdComputer_cfi import *

# Positive-only Tags

# Jet BProb
from RecoBTag.ImpactParameter.positiveOnlyJetBProbabilityComputer_cfi import *
# Jet Prob
from RecoBTag.ImpactParameter.positiveOnlyJetProbabilityComputer_cfi import *


# New candidate based fwk

# Jet BProb
from RecoBTag.ImpactParameter.candidateJetBProbabilityComputer_cfi import *
# Jet Prob
from RecoBTag.ImpactParameter.candidateJetProbabilityComputer_cfi import *
# High Eff
from RecoBTag.ImpactParameter.candidateTrackCounting3D2ndComputer_cfi import *
# High Purity
from RecoBTag.ImpactParameter.candidateTrackCounting3D3rdComputer_cfi import *

# Negative Tags

# Jet BProb
from RecoBTag.ImpactParameter.candidateNegativeOnlyJetBProbabilityComputer_cfi import *
# Jet Prob
from RecoBTag.ImpactParameter.candidateNegativeOnlyJetProbabilityComputer_cfi import *
# High Eff
from RecoBTag.ImpactParameter.candidateNegativeTrackCounting3D2ndComputer_cfi import *
# High Purity
from RecoBTag.ImpactParameter.candidateNegativeTrackCounting3D3rdComputer_cfi import *

# Positive-only Tags

# Jet BProb
from RecoBTag.ImpactParameter.candidatePositiveOnlyJetBProbabilityComputer_cfi import *
# Jet Prob
from RecoBTag.ImpactParameter.candidatePositiveOnlyJetProbabilityComputer_cfi import *
