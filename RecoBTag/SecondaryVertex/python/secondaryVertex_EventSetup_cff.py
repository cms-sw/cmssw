import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoBTau.JetTagComputer.jetTagRecord_cfi import *

from RecoBTag.SecondaryVertex.simpleSecondaryVertex2TrkComputer_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertex3TrkComputer_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexV2Computer_cfi import *
from RecoBTag.SecondaryVertex.ghostTrackComputer_cfi import *

# IVF
from RecoBTag.SecondaryVertex.doubleVertex2TrkComputer_cfi import *

# Negative taggers
from RecoBTag.SecondaryVertex.negativeCombinedSecondaryVertexV2Computer_cfi import *

# Positive taggers
from RecoBTag.SecondaryVertex.positiveCombinedSecondaryVertexV2Computer_cfi import *

# New candidate based fwk
from RecoBTag.SecondaryVertex.candidateSimpleSecondaryVertex2TrkComputer_cfi import *
from RecoBTag.SecondaryVertex.candidateSimpleSecondaryVertex3TrkComputer_cfi import *
from RecoBTag.SecondaryVertex.candidateCombinedSecondaryVertexV2Computer_cfi import *
from RecoBTag.SecondaryVertex.candidateCombinedSecondaryVertexSoftLeptonComputer_cfi import *
from RecoBTag.SecondaryVertex.candidateBoostedDoubleSecondaryVertexAK8Computer_cfi import *
from RecoBTag.SecondaryVertex.candidateBoostedDoubleSecondaryVertexCA15Computer_cfi import *

# Negative taggers
from RecoBTag.SecondaryVertex.candidateNegativeCombinedSecondaryVertexV2Computer_cfi import *

# Positive taggers
from RecoBTag.SecondaryVertex.candidatePositiveCombinedSecondaryVertexV2Computer_cfi import *
