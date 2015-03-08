
import FWCore.ParameterSet.Config as cms

# customize to use upgrade L1 emulation 

from L1Trigger.Configuration.L1Trigger_custom import customiseL1Menu

# customization of run L1 emulator for 2015 run configuration
def customiseSimL1EmulatorForPostLS1(process):
    #print "INFO:  Customising L1T emulator for 2015 run configuration"
    #print "INFO:  Customize the L1 menu"
    # the following line will break HLT if HLT menu is not updated with the corresponding menu
    process=customiseL1Menu(process)
    #print "INFO:  loading RCT LUTs"
    process.load("L1Trigger.L1TCalorimeter.caloStage1RCTLuts_cff")
    if hasattr(process,'L1simulation_step'):
        #print "INFO:  Removing GCT from simulation and adding new Stage 1"
        process.load('L1Trigger.L1TCalorimeter.caloStage1Params_cfi')
        process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_cff')
        process.L1simulation_step.replace(process.simGctDigis,process.L1TCaloStage1)
        process.rctUpgradeFormatDigis.regionTag = cms.InputTag("simRctDigis")
        process.rctUpgradeFormatDigis.emTag = cms.InputTag("simRctDigis")
        #print "New L1 simulation step is:", process.L1simulation_step
        process.simGtDigis.GmtInputTag = 'simGmtDigis'
        process.simGtDigis.GctInputTag = 'caloStage1LegacyFormatDigis'
        process.simGtDigis.TechnicalTriggersInputTags = cms.VInputTag( )
        process.gctDigiToRaw.gctInputLabel = 'caloStage1LegacyFormatDigis'
    return process

#    #
#    # Plan B:  (Not Needed if packing/unpacking of Stage 1 calo via legacy formats and GCT packer works)
#    #
#    process.digi2raw_step.remove(process.gctDigiToRaw)
#   
#    # Carry forward legacy format digis for now (keep rest of workflow working)
#    alist=['RAWSIM','RAWDEBUG','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT']
#    for a in alist:
#        b=a+'output'
#        if hasattr(process,b):
#            getattr(process,b).outputCommands.append('keep *_caloStage1LegacyFormatDigis_*_*')
#            print "INFO:  keeping L1T legacy format digis in event."
#    
#    blist=['l1extraParticles','recoL1ExtraParticles','hltL1ExtraParticles','dqmL1ExtraParticles']
#    for b in blist:
#        if hasattr(process,b):
#            print "INFO:  customizing ", b, "to use simulated legacy formats, without packing/unpacking"
#            getattr(process, b).etTotalSource = cms.InputTag("caloStage1LegacyFormatDigis")
#            getattr(process, b).nonIsolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm")
#            getattr(process, b).etMissSource = cms.InputTag("caloStage1LegacyFormatDigis")
#            getattr(process, b).htMissSource = cms.InputTag("caloStage1LegacyFormatDigis")
#            getattr(process, b).forwardJetSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets")
#            getattr(process, b).centralJetSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets")
#            getattr(process, b).tauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets")
#            getattr(process, b).isolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm")
#            getattr(process, b).etHadSource = cms.InputTag("caloStage1LegacyFormatDigis")
#            getattr(process, b).hfRingEtSumsSource = cms.InputTag("caloStage1LegacyFormatDigis")
#            getattr(process, b).hfRingBitCountsSource = cms.InputTag("caloStage1LegacyFormatDigis")
#
#    # automatic replacements of "simGctDigis" instead of "hltGctDigis"
#    for module in process.__dict__.itervalues():
#        if isinstance(module, cms._Module):
#            for parameter in module.__dict__.itervalues():
#                if isinstance(parameter, cms.InputTag):
#                    if parameter.moduleLabel == 'hltGctDigis':
#                        parameter.moduleLabel = "simGctDigis"

