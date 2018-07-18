# Quality tests for L1 Emulator DQM (L1T)

import FWCore.ParameterSet.Config as cms

# L1 systems quality tests

# Stage 2 L1 Emulator Quality tests
from DQM.L1TMonitorClient.L1TStage2CaloLayer1DEQualityTests_cfi import *
from DQM.L1TMonitorClient.L1TStage2uGTDEQualityTests_cfi import *
from DQM.L1TMonitorClient.L1TStage2uGMTDEQualityTests_cfi import *
from DQM.L1TMonitorClient.L1TStage2BMTFDEQualityTests_cfi import *
from DQM.L1TMonitorClient.L1TStage2OMTFDEQualityTests_cfi import *
from DQM.L1TMonitorClient.L1TStage2EMTFDEQualityTests_cfi import *

# L1 objects quality tests

# sequence for L1 systems
l1TEmulatorSystemQualityTests = cms.Sequence(
                                  l1TStage2CaloLayer1DEQualityTests +
                                  l1TStage2uGTDEQualityTests +
                                  l1TStage2uGMTDEQualityTests +
                                  l1TStage2BMTFDEQualityTests +
                                  l1TStage2OMTFDEQualityTests +
                                  l1TStage2EMTFDEQualityTests
                                  )

# sequence for L1 objects
l1TEmulatorObjectQualityTests = cms.Sequence(
                                  )


# general sequence
l1TStage2EmulatorQualityTests = cms.Sequence(
                                              l1TEmulatorSystemQualityTests + 
                                              l1TEmulatorObjectQualityTests
                                              )
