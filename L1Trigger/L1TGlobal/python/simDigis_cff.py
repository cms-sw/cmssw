import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
#
# Legacy Trigger:
#
if not (eras.stage2L1Trigger.isChosen()):
    print "L1TGlobal Sequence configured for Legacy trigger (Run1 and Run 2015). "
#
# -  Global Trigger emulator
#
    import L1Trigger.GlobalTrigger.gtDigis_cfi
    simGtDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()
    simGtDigis.GmtInputTag = 'simGmtDigis'
    simGtDigis.GctInputTag = 'simGctDigis'
    simGtDigis.TechnicalTriggersInputTags = cms.VInputTag(
        cms.InputTag( 'simBscDigis' ), 
        cms.InputTag( 'simRpcTechTrigDigis' ),
        cms.InputTag( 'simHcalTechTrigDigis' ),
        cms.InputTag( 'simCastorTechTrigDigis' )
        )
    SimL1TGlobal = cms.Sequence(simGtDigis)

#
# Stage-2 Trigger
#
if eras.stage2L1Trigger.isChosen():
#
# -  Global Trigger emulator
#
    print "L1TGlobal Sequence configured for Stage-2 (2016) trigger. "
    from L1Trigger.L1TGlobal.simGtStage2Digis_cfi import *
    simGtStage2Digis.caloInputTag = cms.InputTag('simCaloStage2Digis')
    simGtStage2Digis.GmtInputTag = cms.InputTag('simGmtStage2Digis')
    simGtStage2Digis.PrescaleCSVFile = cms.string('prescale_L1TGlobal.csv')
    simGtStage2Digis.PrescaleSet = cms.uint32(1)
    SimL1TGlobal = cms.Sequence(simGtStage2Digis)
