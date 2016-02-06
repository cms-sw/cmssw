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

# now change all the inputs to the first step of the emulation to use
# unpacked digis instead of simDigis:
simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
    cms.InputTag('hcalDigis'),
    cms.InputTag('hcalDigis')
    )
# not sure what this does...
HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)


# ...
simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigis'
simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'muonCSCDigis', 'MuonCSCComparatorDigi' )
simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'muonCSCDigis', 'MuonCSCWireDigi' )




L1TRerunHCALTP_FromRAW = cms.Sequence(


)


# calo TP generation not normally run during Sim, step, so preprend it to the sequence:
L1TReEmulateFromRAW = cms.Sequence(
    # hcalDigis *
    simHcalTriggerPrimitiveDigis
    * SimL1Emulator 
    )

if not (eras.stage2L1Trigger.isChosen()):
    # HCAL input would be from hcalDigis if hack not needed
    from L1Trigger.Configuration.SimL1Emulator_cff import simRctDigis
    simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalDigis:EcalTriggerPrimitives' ) )
    simRctDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'simHcalTriggerPrimitiveDigis' ) )
    simRpcTriggerDigis.label         = 'muonRPCDigis'
    # simRpcTechTrigDigis.RPCDigiLabel = 'muonRPCDigis' # IGNORING TECH TRIGGERS FOR NOW

if eras.stage2L1Trigger.isChosen():
    simTwinMuxDigis.RPC_Source         = cms.InputTag('muonRPCDigis')
    simOmtfDigis.srcRPC                = cms.InputTag('muonRPCDigis')
    simCaloStage2Layer1Digis.ecalToken = cms.InputTag('ecalDigis:EcalTriggerPrimitives')
    simCaloStage2Layer1Digis.hcalToken = cms.InputTag('simHcalTriggerPrimitiveDigis')
    # this is a hack for -3 BX discrepancy between MC and re-Emulation, not yet understood:
    # simMuonQualityAdjusterDigis.bmtfBxOffset = cms.int32(3) 
    # Picking up simulation a bit further downstream for now:
    simTwinMuxDigis.DTDigi_Source = cms.InputTag("dttfDigis")
    simTwinMuxDigis.DTThetaDigi_Source = cms.InputTag("dttfDigis")
    simBmtfDigis.DTDigi_Source       = cms.InputTag("simTwinMuxDigis")
    simBmtfDigis.DTDigi_Theta_Source = cms.InputTag("dttfDigis")
