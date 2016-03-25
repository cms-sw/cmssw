import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

# this function expects an incomplete list of subsystems to emulate
# and returns a complete list, to ensure all required subsystems are emulated
def getSubsystemsToEmulate(subsys):

    if not eras.stage2L1Trigger.isChosen():
        if 'ECAL' in subsys:
            subsys.append('RCT')
        if 'HCAL' in subsys:
            subsys.append('RCT')
        if 'RCT' in subsys:
            subsys.append('GCT')
        if 'GCT' in subsys:
            subsys.append('GT')
        if 'CSC' in subsys:
            subsys.append('CSCTF1')
        if 'DT' in subsys:
            subsys.append('DTTF')
        if 'DTTF' in subsys:
            subsys.append('CSCTF2')
            subsys.append('GMT')
        if 'CSCTF' in subsys:
            subsys.append('CSCTF1')
            subsys.append('CSCTF2')
            subsys.append('GMT')
        if 'CSCTF2' in subsys:
            subsys.append('GMT')
        if 'RPCTF' in subsys:
            subsys.append('GMT')
        if 'GMT' in subsys:
            subsys.append('GT')

    if eras.stage1L1Trigger.isChosen():
        if 'ECAL' in subsys:
            subsys.append('RCT')
        if 'HCAL' in subsys:
            subsys.append('RCT')
        if 'RCT' in subsys:
            subsys.append('S1CALOL2')
        if 'S1CALOL2' in subsys:
            subsys.append('GT')
        if 'CSC' in subsys:
            subsys.append('CSCTF1')
        if 'DT' in subsys:
            subsys.append('DTTF')
        if 'DTTF' in subsys:
            subsys.append('CSCTF2')
            subsys.append('GMT')
        if 'CSCTF' in subsys:
            subsys.append('CSCTF1')
            subsys.append('CSCTF2')
            subsys.append('GMT')
        if 'CSCTF2' in subsys:
            subsys.append('GMT')
        if 'RPCTF' in subsys:
            subsys.append('GMT')
        if 'GMT' in subsys:
            subsys.append('GT')

    if eras.stage2L1Trigger.isChosen():
        if 'ECAL' in subsys:
            subsys.append('CALOL1')
        if 'HCAL' in subsys:
            subsys.append('CALOL1')
        if 'CALOL1' in subsys:
            subsys.append('CALOL2')
        if 'CALOL2' in subsys:
            subsys.append('GT')
        if 'CSC' in subsys:
            subsys.append('EMTF')
            subsys.append('OMTF')
        if 'DT' in subsys:
            subsys.append('BMTF')
            subsys.append('OMTF')
        if 'RPC' in subsys:
            subsys.append('BMTF')
            subsys.append('EMTF')
            subsys.append('OMTF')
        if 'BMTF' in subsys:
            subsys.append('GMT')
        if 'EMTF' in subsys:
            subsys.append('GMT')
        if 'OMTF' in subsys:
            subsys.append('GMT')
        if 'GMT' in subsys:
            subsys.append('GT')

    out = []
    for sys in subsys:
        if sys not in out:
            out.append(sys)

    return out


# this method sets the input tags of each system to be emulated
# depending on whether the upstream system is set to be emulated or not
# assumption is starting from RAW, not (SIM)DIGI !
# if starting from (SIM)DIGI, no changes in input tags required
def setInputTags(process, subsys):

    if 'ECAL' in subsys:
        process.simEcalTriggerPrimitiveDigis.inputLabel = cms.InputTag( 'ecalDigis' )

    if 'HCAL' in subsys:
        process.simHcalTriggerPrimitiveDigis.inputLabel = cms.InputTag( 'hcalDigis' )

    if 'RCT' in subsys:
        if 'ECAL' not in subsys:
            process.simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalDigis:ECALTriggerPrimitives' ) )
        if 'HCAL' not in subsys:
            process.simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'hcalDigis' ) )

    if ('GCT' in subsys) and ('RCT' not in subsys):
        process.simGctDigis.inputLabel = 'gctDigis'

    if 'DT' in subsys:
        process.simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigis'

    if 'CSC' in subsys:
        process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'muonCSCDigis', 'MuonCSCComparatorDigi' )
        process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'muonCSCDigis', 'MuonCSCWireDigi' )

    if 'DTTF' in subsys:
        if 'DT' not in subsys:
            process.simDttfDigis.DTDigi_Source  = 'dtTriggerPrimitiveDigis'
        if 'CSCTF1' not in subsys:
            process.simDttfDigis.CSCStub_Source = 'csctfDigis'

    if 'CSCTF1' in subsys:
        if 'CSC' not in subsys:
            process.simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag( 'cscTriggerPrimitiveDigis', 'MPCSORTED' )
        if 'DT' not in subsys:
            process.simCsctfTrackDigis.DTproducer = 'dtTriggerPrimitiveDigis'

    if 'CSCTF2' in subsys and 'CSCTF1' not in subsys:
        process.simCsctfDigis.CSCTrackProducer = 'csctfDigis'

    if 'RPCTF' in subsys:
        process.simRpcTriggerDigis.label = 'muonRPCDigis'

    if 'GMT' in subsys:
        if 'DTTF' not in subsys:
            process.simGmtDigis.DTCandidates   = cms.InputTag( 'dttfDigis', 'DT' )
        if 'CSCTF2' not in subsys:
            process.simGmtDigis.CSCCandidates  = cms.InputTag( 'csctfDigis', 'CSC' )
        if 'RPCTF' not in subsys:
            process.simGmtDigis.RPCbCandidates = cms.InputTag( 'rpcTriggerDigis', 'RPCb' )
            process.simGmtDigis.RPCfCandidates = cms.InputTag( 'rpcTriggerDigis', 'RPCf' )

    if 'S1CALOL2' in subsys:
        if 'RCT' not in subsys:
            process.simRctUpgradeFormatDigis.regionTag = cms.InputTag("caloStage1Digis")
            process.simRctUpgradeFormatDigis.emTag = cms.InputTag("caloStage1Digis")

    if 'CALOL1' in subsys:
        if 'ECAL' not in subsys:
            process.ecalToken = cms.InputTag("ecalDigis:EcalTriggerPrimitiveDigis")
        if 'HCAL' not in subsys:
            process.hcalToken = cms.InputTag("hcalDigis")

    if 'CALOL2' in subsys:
        if 'CALOL1' not in subsys:
            process.simCaloStage2Digis.towerToken = cms.InputTag("caloStage2Digis", "CaloTower")


def modifySimL1EmulatorForReEmulation(SimL1Emulator_object, subsys=[]):

    if not eras.stage2L1Trigger.isChosen():
#        if 'ECAL' not in subsys:
#            digiSeq_object.remove(simEcalTriggerPrimitiveDigis)
#        if 'HCAL' not in subsys:
#            digiSeq_object.remove(simHcalTriggerPrimitiveDigis)
        if 'RCT' not in subsys:
            SimL1Emulator_object.remove(simRctDigis)
        if 'GCT' not in subsys:
            SimL1Emulator_object.remove(simGctDigis)
        if 'CSC' not in subsys:
            SimL1Emulator_object.remove(simCscTriggerPrimitiveDigis)
        if 'DT' not in subsys:
            SimL1Emulator_object.remove(simDtTriggerPrimitiveDigis)
        if 'CSCTF1' not in subsys:
            SimL1Emulator_object.remove(simCsctfTrackDigis)
        if 'CSCTF2' not in subsys:
            SimL1Emulator_object.remove(simCsctfDigis)
        if 'DTTF' not in subsys:
            SimL1Emulator_object.remove(simDttfDigis)
        if 'RPCTF' not in subsys:
            SimL1Emulator_object.remove(simRpcTriggerDigis)
        if 'GMT' not in subsys:
            SimL1Emulator_object.remove(simGmtDigis)
        if 'GT' not in subsys:
            SimL1Emulator_object.remove(simGtDigis)

    if eras.stage1L1Trigger.isChosen():
        if 'S1CALOL2' not in subsys:
            SimL1Emulator_object.remove(simRctUpgradeFormatDigis)
            SimL1Emulator_object.remove(simCaloStage1Digis)
            SimL1Emulator_object.remove(simCaloStage1FinalDigis)
            SimL1Emulator_object.remove(simCaloStage1LegacyFormatDigis)

    if eras.stage1L1Trigger.isChosen():
        if 'CALOL1' not in subsys:
            SimL1Emulator_object.remove(simCaloStage2Layer1Digis)
        if 'CALOL2' not in subsys:
            SimL1Emulator_object.remove(simCaloStage2Digis)


def customise_ReEmulateL1(process, subsys=[]):

    subsysFull = getSubsystemsToEmulate(subsys)
  
    modifySimL1EmulatorForReEmulation(process.SimL1Emulator, subsysFull)



def test():
    print "Testing"
    test = ['ECAL', 'DTTF']
    print "Subsys in : ", test
    print "Subsys out : ", getSubsystemsToEmulate(test, 'legacy')
    test = ['ECAL', 'BMTF']
    print "Subsys in : ", test
    print "Subsys out : ", getSubsystemsToEmulate(test, 'stage2')
