# Quality tests for L1 Trigger DQM (L1T)

import FWCore.ParameterSet.Config as cms

# L1 systems quality tests

# Stage 2 L1 Trigger Quality tests
from DQM.L1TMonitorClient.L1TStage2CaloLayer1QualityTests_cfi import *

# L1 objects quality tests

# sequence for L1 systems
l1TriggerSystemQualityTests = cms.Sequence(
                                l1TStage2CaloLayer1QualityTests
                                )

# sequence for L1 objects
l1TriggerObjectQualityTests = cms.Sequence(
                                )


# general sequence
l1TStage2QualityTests = cms.Sequence(
                                      l1TriggerSystemQualityTests + 
                                      l1TriggerObjectQualityTests
                                      )

