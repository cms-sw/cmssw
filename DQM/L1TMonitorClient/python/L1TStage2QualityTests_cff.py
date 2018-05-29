# Quality tests for L1 Trigger DQM (L1T)

import FWCore.ParameterSet.Config as cms

# L1 systems quality tests

# Stage 2 L1 Trigger Quality tests
from DQM.L1TMonitorClient.L1TStage2CaloLayer1QualityTests_cfi import *
from DQM.L1TMonitorClient.L1TStage2uGMTQualityTests_cfi import *
from DQM.L1TMonitorClient.L1TStage2BMTFQualityTests_cfi import *
from DQM.L1TMonitorClient.L1TStage2OMTFQualityTests_cfi import *
from DQM.L1TMonitorClient.L1TStage2EMTFQualityTests_cfi import *
from DQM.L1TMonitorClient.L1TStage2uGTQualityTests_cfi import *
from DQM.L1TMonitorClient.L1TStage2MuonQualityTests_cff import *

# L1 objects quality tests

# sequence for L1 systems
l1TriggerSystemQualityTests = cms.Sequence(
    l1TStage2CaloLayer1QualityTests +
    l1TStage2uGMTQualityTests +
    l1TStage2BMTFQualityTests +
    l1TStage2OMTFQualityTests +
    l1TStage2EMTFQualityTests +
    l1TStage2uGTQualityTests
)

# sequence for L1 objects
l1TriggerObjectQualityTests = cms.Sequence(
    l1TStage2MuonQualityTests +
    l1TStage2MuonQualityTestsCollisions
)

# general sequence
l1TStage2QualityTests = cms.Sequence(
    l1TriggerSystemQualityTests +
    l1TriggerObjectQualityTests
)

