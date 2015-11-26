# defines:
#  
#     L1TReEmulateFromRAW = cms.Sequence(...)
#
# properly configured for the current Era (e.g. Run1, 2015, or 2016).  Also
# configures event setup producers appropriate to the current Era, to handle
# conditions which are not yet available in the GT.

import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from L1Trigger.Configuration.CaloTriggerPrimitives_cff import *
from L1Trigger.Configuration.SimL1Emulator_cff import *

# Legacy trigger primitive emulations still running in 2016 trigger:
#
# NOTE:  2016 HCAL HF TPs require a new emulation, which is not yet available...
#

# calo TP generation not normally run during Sim, step, so preprend it to the sequence:
L1TReEmulateFromRAW = cms.Sequence(
    CaloTriggerPrimitives +
    SimL1Emulator 
    )

# now simple change all the inputs to the first step of the emulation to use
# unpacked digis instead of simDigis:
simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
    cms.InputTag('hcalDigis'),
    cms.InputTag('hcalDigis')
    )
simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigis'
simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'muonCSCDigis', 'MuonCSCComparatorDigi' )
simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'muonCSCDigis', 'MuonCSCWireDigi' )

if not (eras.stage2L1Trigger.isChosen()):
    simRpcTriggerDigis.label         = 'muonRPCDigis'
    # simRpcTechTrigDigis.RPCDigiLabel = 'muonRPCDigis' # IGNORING TECH TRIGGERS FOR NOW

if eras.stage2L1Trigger.isChosen():
    simTwinMuxDigis.RPC_Source     = cms.InputTag('muonRPCDigis')
    simOmtfDigis.srcRPC            = cms.InputTag('muonRPCDigis')
