
import FWCore.ParameterSet.Config as cms


# customize to use upgrade L1 emulation

from L1Trigger.Configuration.L1Trigger_custom import customiseL1Menu

# customization of run L1 emulator for 2015 run configuration
def customiseSimL1EmulatorForPostLS1_nomenu(process):
    #print "INFO:  Customising L1T emulator for 2015 run configuration"
    #print "INFO:  Customize the L1 menu"
    # the following line will break HLT if HLT menu is not updated with the corresponding menu
    #process=customiseL1Menu(process)
    #print "INFO:  loading RCT LUTs"
    process.load("L1Trigger.L1TCalorimeter.caloStage1RCTLuts_cff")

    process.load("L1Trigger.L1TCommon.l1tDigiToRaw_cfi")
    process.load("L1Trigger.L1TCommon.l1tRawToDigi_cfi")
    process.load("L1Trigger.L1TCommon.caloStage1LegacyFormatDigis_cfi")


    if hasattr(process,'L1simulation_step'):
        #print "INFO:  Removing GCT from simulation and adding new Stage 1"
        process.load('L1Trigger.L1TCalorimeter.caloStage1Params_cfi')
        process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_cff')
        process.L1simulation_step.replace(process.simGctDigis,process.L1TCaloStage1)
        #process.rctUpgradeFormatDigis.regionTag = cms.InputTag("simRctDigis")
        #process.rctUpgradeFormatDigis.emTag = cms.InputTag("simRctDigis")
        #print "New L1 simulation step is:", process.L1simulation_step
        process.simGtDigis.GmtInputTag = 'simGmtDigis'
        process.simGtDigis.GctInputTag = 'simCaloStage1LegacyFormatDigis'
        process.simGtDigis.TechnicalTriggersInputTags = cms.VInputTag( )
        process.gctDigiToRaw.gctInputLabel = 'simCaloStage1LegacyFormatDigis'
    if hasattr(process, 'DigiToRaw'):
        #print "INFO:  customizing DigiToRaw for Stage 1"
        #print process.DigiToRaw
        process.l1tDigiToRaw.InputLabel = cms.InputTag("simCaloStage1FinalDigis", "")
        process.l1tDigiToRaw.TauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "rlxTaus")
        process.l1tDigiToRaw.IsoTauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "isoTaus")
        process.l1tDigiToRaw.HFBitCountsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFBitCounts")
        process.l1tDigiToRaw.HFRingSumsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFRingSums")
        process.l1tDigiToRawSeq = cms.Sequence(process.gctDigiToRaw + process.l1tDigiToRaw);
        process.DigiToRaw.replace(process.gctDigiToRaw, process.l1tDigiToRawSeq)
        #print process.DigiToRaw
        if hasattr(process, 'rawDataCollector'):
            #print "INFO:  customizing rawDataCollector for Stage 1"
            process.rawDataCollector.RawCollectionList.append(cms.InputTag("l1tDigiToRaw"))
    if hasattr(process, 'RawToDigi'):
        #print "INFO:  customizing L1RawToDigi for Stage 1"
        #print process.RawToDigi
        process.L1RawToDigiSeq = cms.Sequence(process.gctDigis+process.caloStage1Digis+process.caloStage1LegacyFormatDigis)
        process.RawToDigi.replace(process.gctDigis, process.L1RawToDigiSeq)
        #print process.RawToDigi


#hltGtDigis+hltGctDigis+hltL1GtObjectMap+hltL1extraParticles
#csctfDigis+dttfDigis+gctDigis+caloStage1Digis+caloStage1LegacyFormatDigis+gtDigis+gtEvmDigis+siPixelDigis+siStripDigis+ecalDigis+ecalPreshowerDigis+hcalDigis+muonCSCDigis+muonDTDigis+muonRPCDigis+castorDigis+scalersRawToDigi+hltL1extraParticles
    if hasattr(process, 'HLTL1UnpackerSequence'):
        #print "INFO: customizing HLTL1UnpackerSequence for Stage 1"
        #print process.HLTL1UnpackerSequence

        # extend sequence to add Layer 1 unpacking and conversion back to legacy format
        process.hltCaloStage1Digis = process.caloStage1Digis.clone()
        process.hltCaloStage1LegacyFormatDigis = process.caloStage1LegacyFormatDigis.clone()
        process.hltCaloStage1LegacyFormatDigis.InputCollection = cms.InputTag("hltCaloStage1Digis")
        process.hltCaloStage1LegacyFormatDigis.InputRlxTauCollection = cms.InputTag("hltCaloStage1Digis:rlxTaus")
        process.hltCaloStage1LegacyFormatDigis.InputIsoTauCollection = cms.InputTag("hltCaloStage1Digis:isoTaus")
        process.hltCaloStage1LegacyFormatDigis.InputHFSumsCollection = cms.InputTag("hltCaloStage1Digis:HFRingSums")
        process.hltCaloStage1LegacyFormatDigis.InputHFCountsCollection = cms.InputTag("hltCaloStage1Digis:HFBitCounts")
        #process.hltL1RawToDigiSeq = cms.Sequence(process.hltGctDigis+process.hltCaloStage1 + process.hltCaloStage1LegacyFormatDigis)
        process.hltL1RawToDigiSeq = cms.Sequence(process.hltCaloStage1Digis + process.hltCaloStage1LegacyFormatDigis)
        process.HLTL1UnpackerSequence.replace(process.hltGctDigis, process.hltL1RawToDigiSeq)


        # Alternate (possibly necessary approach) not yet debugged...
        # redefine the HLTL1UnpackerSequence
        #HLTL1UnpackerSequence = cms.Sequence( process.hltGtDigis+process.hltGctDigis+process.hltL1GtObjectMap+process.hltL1extraParticles + process.hltL1extraParticles )
        #HLTL1UnpackerSequence = cms.Sequence( process.hltGtDigis++process.hltL1GtObjectMap+process.hltL1extraParticles + process.hltL1extraParticles )
        #process.load("L1Trigger.L1TCommon.l1tRawToDigi_cfi")
        #process.l1tRawToDigi.Setup = cms.string("stage1::CaloSetup")
        #process.load("L1Trigger.L1TCommon.caloStage1LegacyFormatDigis_cfi")
        #process.L1RawToDigiSeq = cms.Sequence(process.gctDigis+process.caloStage1Digis+process.caloStage1LegacyFormatDigis)
        #for iterable in process.sequences.itervalues():
        #    iterable.replace( process.HLTL1UnpackerSequence, HLTL1UnpackerSequence)

        #for iterable in process.paths.itervalues():
        #    iterable.replace( process.HLTL1UnpackerSequence, HLTL1UnpackerSequence)

        #for iterable in process.endpaths.itervalues():
        #    iterable.replace( process.HLTL1UnpackerSequence, HLTL1UnpackerSequence)

        #process.HLTL1UnpackerSequence = HLTL1UnpackerSequence
        #print process.HLTL1UnpackerSequence

    alist=['hltL1extraParticles']
    for a in alist:
        #print "INFO: checking for", a, "in process."
        if hasattr(process,a):
            #print "INFO:  customizing ", a, "to use new calo Stage 1 digis converted to legacy format"
            getattr(process, a).etTotalSource = cms.InputTag("hltCaloStage1LegacyFormatDigis")
            getattr(process, a).nonIsolatedEmSource = cms.InputTag("hltCaloStage1LegacyFormatDigis","nonIsoEm")
            getattr(process, a).etMissSource = cms.InputTag("hltCaloStage1LegacyFormatDigis")
            getattr(process, a).htMissSource = cms.InputTag("hltCaloStage1LegacyFormatDigis")
            getattr(process, a).forwardJetSource = cms.InputTag("hltCaloStage1LegacyFormatDigis","forJets")
            getattr(process, a).centralJetSource = cms.InputTag("hltCaloStage1LegacyFormatDigis","cenJets")
            getattr(process, a).tauJetSource = cms.InputTag("hltCaloStage1LegacyFormatDigis","tauJets")
            getattr(process, a).isoTauJetSource = cms.InputTag("hltCaloStage1LegacyFormatDigis","isoTauJets")
            getattr(process, a).isolatedEmSource = cms.InputTag("hltCaloStage1LegacyFormatDigis","isoEm")
            getattr(process, a).etHadSource = cms.InputTag("hltCaloStage1LegacyFormatDigis")
            getattr(process, a).hfRingEtSumsSource = cms.InputTag("hltCaloStage1LegacyFormatDigis")
            getattr(process, a).hfRingBitCountsSource = cms.InputTag("hltCaloStage1LegacyFormatDigis")

    blist=['l1extraParticles','recoL1extraParticles','dqmL1ExtraParticles']
    for b in blist:
        #print "INFO: checking for", b, "in process."
        if hasattr(process,b):
            #print "INFO:  customizing ", b, "to use new calo Stage 1 digis converted to legacy format"
            getattr(process, b).etTotalSource = cms.InputTag("caloStage1LegacyFormatDigis")
            getattr(process, b).nonIsolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm")
            getattr(process, b).etMissSource = cms.InputTag("caloStage1LegacyFormatDigis")
            getattr(process, b).htMissSource = cms.InputTag("caloStage1LegacyFormatDigis")
            getattr(process, b).forwardJetSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets")
            getattr(process, b).centralJetSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets")
            getattr(process, b).tauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets")
            getattr(process, b).isoTauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","isoTauJets")
            getattr(process, b).isolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm")
            getattr(process, b).etHadSource = cms.InputTag("caloStage1LegacyFormatDigis")
            getattr(process, b).hfRingEtSumsSource = cms.InputTag("caloStage1LegacyFormatDigis")
            getattr(process, b).hfRingBitCountsSource = cms.InputTag("caloStage1LegacyFormatDigis")


#    # THIS IS A HACK FOR NOW, WHILE WE SORT OUT FED ID:
#    clist=['RAWSIM','RAWDEBUG','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT']
#    for c in clist:
#        b=c+'output'
#        if hasattr(process,b):
#            getattr(process,b).outputCommands.append('keep *_l1tDigiToRaw_*_*')
#            print "INFO:  keeping l1tDigiToRaw output in the event."


#    process.MessageLogger = cms.Service(
#        "MessageLogger",
#        destinations   = cms.untracked.vstring(
#            'detailedInfo',
#            'critical'
#            ),
#        detailedInfo   = cms.untracked.PSet(
#            threshold  = cms.untracked.string('DEBUG')
#            ),
#        debugModules = cms.untracked.vstring(
#            'l1tDigiToRaw', 'l1tRawToDigi'
#            )
#        )
#    print process.HLTSchedule
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
#
#    # automatic replacements of "simGctDigis" instead of "hltGctDigis"
#    for module in process.__dict__.itervalues():
#        if isinstance(module, cms._Module):
#            for parameter in module.__dict__.itervalues():
#                if isinstance(parameter, cms.InputTag):
#                    if parameter.moduleLabel == 'hltGctDigis':
#                        parameter.moduleLabel = "simGctDigis"

def customiseSimL1EmulatorForPostLS1(process):
    process=customiseL1Menu(process)
    process=customiseSimL1EmulatorForPostLS1_nomenu(process)
    return process

def customiseSimL1EmulatorForPostLS1_HI(process):
    process=customiseL1Menu_HI(process)
    process=customiseSimL1EmulatorForPostLS1_nomenu(process)
    return process
