import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# DQM Modules

# CaloLayer1
from DQM.L1TMonitor.L1TStage2CaloLayer1_cfi import *

# CaloLayer2
from DQM.L1TMonitor.L1TStage2CaloLayer2_cfi import *

# BMTF
from DQM.L1TMonitor.L1TStage2BMTF_cff import *

# OMTF
from DQM.L1TMonitor.L1TStage2OMTF_cfi import *

# CPPF
from DQM.L1TMonitor.L1TStage2CPPF_cfi import *

# EMTF
from DQM.L1TMonitor.L1TStage2EMTF_cfi import *

# Shower
from DQM.L1TMonitor.L1TStage2RegionalShower_cfi import *

# uGMT
from DQM.L1TMonitor.L1TStage2uGMT_cff import *

#map for online objects
from DQM.L1TMonitor.L1TObjectsTiming_cfi import *

# uGT
from DQM.L1TMonitor.L1TStage2uGT_cff import *

#-------------------------------------------------
# Stage2 Unpacking and DQM Sequences

# sequence to run for every event
l1tStage2OnlineDQM = cms.Sequence(
    l1tStage2CaloLayer1 +
    l1tStage2CaloLayer2 +
    l1tStage2BmtfOnlineDQMSeq +
    l1tStage2Omtf +
    l1tStage2Emtf +
    l1tStage2Cppf +
    l1tStage2RegionalShower +  
    l1tStage2uGMTOnlineDQMSeq +
    l1tObjectsTiming +
    l1tStage2uGTOnlineDQMSeq
)

# sequence to run only for validation events
l1tStage2OnlineDQMValidationEvents = cms.Sequence(
    l1tStage2BmtfValidationEventOnlineDQMSeq +
    l1tStage2uGMTValidationEventOnlineDQMSeq +
    l1tStage2uGTValidationEventOnlineDQMSeq
)

