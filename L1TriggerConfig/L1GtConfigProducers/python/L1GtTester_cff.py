# cff file to print all the L1 GT records
#
# V M Ghete  2008 - 2010 - 2012

import FWCore.ParameterSet.Config as cms

# L1 GT stable parameters
from L1TriggerConfig.L1GtConfigProducers.l1GtStableParametersTester_cfi import *

# L1 GT parameters
from L1TriggerConfig.L1GtConfigProducers.l1GtParametersTester_cfi import *

# L1 GT board maps
from L1TriggerConfig.L1GtConfigProducers.l1GtBoardMapsTester_cfi import *

# L1 GT PSB setup
from L1TriggerConfig.L1GtConfigProducers.l1GtPsbSetupTester_cfi import *

# prescale factors and masks
from L1TriggerConfig.L1GtConfigProducers.l1GtPrescaleFactorsAndMasksTester_cfi import *

#
from L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuTester_cfi import *

seqL1GtStableParameters = cms.Sequence(l1GtStableParametersTester)
seqL1GtParameters = cms.Sequence(l1GtParametersTester)
seqL1GtBoardMaps = cms.Sequence(l1GtBoardMapsTester)
seqL1GtPsbSetup = cms.Sequence(l1GtPsbSetupTester)
seqL1GtPrescaleFactorsAndMasks = cms.Sequence(l1GtPrescaleFactorsAndMasksTester)
seqL1GtTriggerMenu = cms.Sequence(l1GtTriggerMenuTester)

# define the sequence to be use in L1GlobalTagTest_cff

printGlobalTagL1Gt = cms.Sequence(seqL1GtStableParameters
                                  *seqL1GtParameters
                                  *seqL1GtBoardMaps
                                  *seqL1GtPsbSetup
                                  *seqL1GtPrescaleFactorsAndMasks
                                  *seqL1GtTriggerMenu
                                  )



