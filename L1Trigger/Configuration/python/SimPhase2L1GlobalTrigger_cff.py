# Make Phase 2 Global Trigger "book end" configurations available as an L1 Configuration file
# To be made available later in Configuration/StandardSequences.
# It's a little circuitous, I admit, but seems correct to be available _everywhere_
# Written by: Andrew Loeliger

import FWCore.ParameterSet.Config as cms

# Get the GT Producer/first of the "book end"s, responsible for GT inputs
from L1Trigger.Phase2L1GT.l1tGTProducer_cff import l1tGTProducer

l1tGTProducerSequence = cms.Sequence(
    l1tGTProducer
)

# Get the Algo Block/second of the "book end"s, responsible for trigger results
from L1Trigger.Phase2L1GT.l1tGTAlgoBlockProducer_cff import l1tGTAlgoBlockProducer

l1tGTAlgoBlockProducerSequence = cms.Sequence(
    l1tGTAlgoBlockProducer
)