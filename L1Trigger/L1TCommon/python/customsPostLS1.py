
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
    #process.load("L1Trigger.L1TCalorimeter.caloStage1RCTLuts_cff")

    process.load("L1Trigger.L1TCommon.l1tDigiToRaw_cfi")
    process.load("L1Trigger.L1TCommon.l1tRawToDigi_cfi")
    process.load("L1Trigger.L1TCommon.caloStage1LegacyFormatDigis_cfi")

    process.load('L1Trigger.L1TCalorimeter.caloStage1Params_cfi')
    process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_cff')

    if hasattr(process, 'simGtDigis'):
        process.simGtDigis.GmtInputTag = 'simGmtDigis'
        process.simGtDigis.GctInputTag = 'simCaloStage1LegacyFormatDigis'
        process.simGtDigis.TechnicalTriggersInputTags = cms.VInputTag( )
    if hasattr(process, 'gctDigiToRaw'):
        process.gctDigiToRaw.gctInputLabel = 'simCaloStage1LegacyFormatDigis'

    if hasattr(process, 'simGctDigis'):
        for sequence in process.sequences:
            #print "INFO:  checking sequence ", sequence
            #print "BEFORE:  ", getattr(process,sequence)
            getattr(process,sequence).replace(process.simGctDigis,process.L1TCaloStage1)
            #print "AFTER:  ", getattr(process,sequence)
        for path in process.paths:
            #print "INFO:  checking path ", path
            #print "BEFORE:  ", getattr(process,path)
            getattr(process,path).replace(process.simGctDigis,process.L1TCaloStage1)
            #print "AFTER:  ", getattr(process,path)

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

    alist=['hltL1GtObjectMap']
    for a in alist:
        #print "INFO: checking for", a, "in process."
        if hasattr(process,a):
            #print "INFO: customizing ", a, "to use new calo Stage 1 digis converted to legacy format"
            getattr(process, a).GctInputTag = cms.InputTag("hltCaloStage1LegacyFormatDigis")

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
            #print "BEFORE:  ", getattr(process, b).centralJetSource
            if (getattr(process, b).centralJetSource == cms.InputTag("simGctDigis","cenJets")):
                getattr(process, b).etTotalSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
                getattr(process, b).nonIsolatedEmSource = cms.InputTag("simCaloStage1LegacyFormatDigis","nonIsoEm")
                getattr(process, b).etMissSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
                getattr(process, b).htMissSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
                getattr(process, b).forwardJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","forJets")
                getattr(process, b).centralJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","cenJets")
                getattr(process, b).tauJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","tauJets")
                getattr(process, b).isoTauJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","isoTauJets")
                getattr(process, b).isolatedEmSource = cms.InputTag("simCaloStage1LegacyFormatDigis","isoEm")
                getattr(process, b).etHadSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
                getattr(process, b).hfRingEtSumsSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
                getattr(process, b).hfRingBitCountsSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
            else:
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
            #print "AFTER:  ", getattr(process, b).centralJetSource

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
