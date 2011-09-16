#
#  Load common sequences
#
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1RequestPhAlgos = hltLevel1GTSeed.clone()
# False allows to read directly from L1 instead fo candidate ObjectMap
l1RequestPhAlgos.L1UseL1TriggerObjectMaps = cms.bool(False)
    #
    # option used forL1UseL1TriggerObjectMaps = False only
    # number of BxInEvent: 1: L1A=0; 3: -1, L1A=0, 1; 5: -2, -1, L1A=0, 1, 
# online is used 5
l1RequestPhAlgos.L1NrBxInEvent = cms.int32(5)

# Request the or of the following bits: from 54 to 62 and 106-107

l1RequestPhAlgos.L1SeedsLogicalExpression = cms.string(
    'L1_SingleMuBeamHalo OR L1_SingleMuOpen OR L1_SingleMu3 OR L1_SingleMu7 OR L1_SingleMu10 OR L1_SingleMu20 OR L1_DoubleMu3')

l1MuBitsSkimseq = cms.Sequence(l1RequestPhAlgos)
