
import FWCore.ParameterSet.Config as cms


# customize to use upgrade L1 emulation

from L1Trigger.Configuration.L1Trigger_custom import customiseL1Menu

# customization of run L1 emulator for 2015 Stage 1 configuration
def customiseSimL1EmulatorForStage1(process):

    process.load("L1Trigger.L1TCommon.l1tDigiToRaw_cfi")
    process.load("EventFilter.L1TRawToDigi.caloStage1Digis_cfi")
    process.load("L1Trigger.L1TCommon.caloStage1LegacyFormatDigis_cfi")

    process.load('L1Trigger.L1TCalorimeter.caloStage1Params_cfi')
    process.load('L1Trigger.L1TCalorimeter.caloConfigStage1PP_cfi')
    process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_cff')

    if hasattr(process, 'simGtDigis'):
        process.simGtDigis.GmtInputTag = 'simGmtDigis'
        process.simGtDigis.GctInputTag = 'simCaloStage1LegacyFormatDigis'
        process.simGtDigis.TechnicalTriggersInputTags = cms.VInputTag( )
    if hasattr(process, 'gctDigiToRaw'):
        process.gctDigiToRaw.gctInputLabel = 'simCaloStage1LegacyFormatDigis'

    if hasattr(process, 'simGctDigis'):
        for sequence in process.sequences:
            getattr(process,sequence).replace(process.simGctDigis,process.L1TCaloStage1)
        for path in process.paths:
            getattr(process,path).replace(process.simGctDigis,process.L1TCaloStage1)

    if hasattr(process, 'DigiToRaw'):
        process.l1tDigiToRaw.InputLabel = cms.InputTag("simCaloStage1FinalDigis", "")
        process.l1tDigiToRaw.TauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "rlxTaus")
        process.l1tDigiToRaw.IsoTauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "isoTaus")
        process.l1tDigiToRaw.HFBitCountsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFBitCounts")
        process.l1tDigiToRaw.HFRingSumsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFRingSums")
        process.l1tDigiToRawSeq = cms.Sequence(process.gctDigiToRaw + process.l1tDigiToRaw);
        process.DigiToRaw.replace(process.gctDigiToRaw, process.l1tDigiToRawSeq)
        if hasattr(process, 'rawDataCollector'):
            process.rawDataCollector.RawCollectionList.append(cms.InputTag("l1tDigiToRaw"))
    if hasattr(process, 'RawToDigi'):
        process.L1RawToDigiSeq = cms.Sequence(process.gctDigis+process.caloStage1Digis+process.caloStage1LegacyFormatDigis)
        process.RawToDigi.replace(process.gctDigis, process.L1RawToDigiSeq)

    blist=['l1extraParticles','recoL1extraParticles','dqmL1ExtraParticles']
    for b in blist:
        if hasattr(process,b):
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

    return process


# customization of run L1 emulator for 2015 Stage 1 configuration
def customiseL1RecoForStage1(process):

    process.load("L1Trigger.L1TCommon.l1tRawToDigi_cfi")
    process.load("L1Trigger.L1TCommon.caloStage1LegacyFormatDigis_cfi")

    if hasattr(process, 'RawToDigi'):
        process.L1RawToDigiSeq = cms.Sequence(process.gctDigis+process.caloStage1Digis+process.caloStage1LegacyFormatDigis)
        process.RawToDigi.replace(process.gctDigis, process.L1RawToDigiSeq)

    blist=['l1extraParticles','recoL1extraParticles','dqmL1ExtraParticles']
    for b in blist:
        if hasattr(process,b):
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

    return process


from L1Trigger.Configuration.customise_overwriteL1Menu import *

def customiseSimL1EmulatorForPostLS1_lowPU(process):
    # load the Stage 1 configuration
    process = customiseSimL1EmulatorForStage1(process)
    # move to the lowPU v3 L1 menu once the HLT has been updated accordingly
    process = L1Menu_Collisions2015_lowPU_v3(process)
    return process

def customiseSimL1EmulatorForPostLS1_50ns(process):
    # load the Stage 1 configuration
    process = customiseSimL1EmulatorForStage1(process)
    # move to the 50ns v2 L1 menu once the HLT has been updated accordingly
    process = L1Menu_Collisions2015_50ns_v2(process)
    return process

def customiseSimL1EmulatorForPostLS1_25ns(process):
    # load the Stage 1 configuration
    process = customiseSimL1EmulatorForStage1(process)
    # load the 25ns v2 L1 menu
    process = L1Menu_Collisions2015_25ns_v2(process)
    return process

# additional customizations needed for HI:
# -> no L1 Menu added here
# -> common post LS1 customizations not called here
def customiseSimL1EmulatorForPostLS1_Additional_HI(process):
    # set the Stage 1 heavy ions-specific parameters
    # all of these should eventually end up in a GT
    process.load('L1Trigger.L1TCalorimeter.caloStage1Params_HI_cfi')
    if hasattr(process,'caloConfig'):
        process.caloConfig.fwVersionLayer2 = cms.uint32(1)
    return process

# dustomization for offline DQM
def customiseStage1ForOfflineDQM(process):

    process.load('DQMOffline.L1Trigger.L1TriggerDqmOffline_cff')

    if hasattr(process, 'l1tMonitorStage1Online'):
        process.l1tRct.rctSource = 'caloStage1Digis'
        process.l1tRctfromRCT.rctSource = 'rctDigis'
        process.l1tPUM.regionSource = cms.InputTag("rctDigis")
        process.l1tStage1Layer2.stage1_layer2_ = cms.bool(True)
        process.l1tStage1Layer2.gctCentralJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets")
        process.l1tStage1Layer2.gctForwardJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets")
        process.l1tStage1Layer2.gctTauJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets")
        process.l1tStage1Layer2.gctIsoTauJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","isoTauJets")
        process.l1tStage1Layer2.gctEnergySumsSource = cms.InputTag("caloStage1LegacyFormatDigis")
        process.l1tStage1Layer2.gctIsoEmSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm")
        process.l1tStage1Layer2.gctNonIsoEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm")
        process.dqmL1ExtraParticlesStage1.etTotalSource = 'caloStage1LegacyFormatDigis'
        process.dqmL1ExtraParticlesStage1.nonIsolatedEmSource = 'caloStage1LegacyFormatDigis:nonIsoEm'
        process.dqmL1ExtraParticlesStage1.etMissSource = 'caloStage1LegacyFormatDigis'
        process.dqmL1ExtraParticlesStage1.htMissSource = 'caloStage1LegacyFormatDigis'
        process.dqmL1ExtraParticlesStage1.forwardJetSource = 'caloStage1LegacyFormatDigis:forJets'
        process.dqmL1ExtraParticlesStage1.centralJetSource = 'caloStage1LegacyFormatDigis:cenJets'
        process.dqmL1ExtraParticlesStage1.tauJetSource = 'caloStage1LegacyFormatDigis:tauJets'
        process.dqmL1ExtraParticlesStage1.isolatedEmSource = 'caloStage1LegacyFormatDigis:isoEm'
        process.dqmL1ExtraParticlesStage1.etHadSource = 'caloStage1LegacyFormatDigis'
        process.dqmL1ExtraParticlesStage1.hfRingEtSumsSource = 'caloStage1LegacyFormatDigis'
        process.dqmL1ExtraParticlesStage1.hfRingBitCountsSource = 'caloStage1LegacyFormatDigis'
        process.l1ExtraDQMStage1.stage1_layer2_ = cms.bool(True)
        process.l1ExtraDQMStage1.L1ExtraIsoTauJetSource_ = cms.InputTag("dqmL1ExtraParticlesStage1", "IsoTau")

    if hasattr(process, 'l1Stage1HwValEmulatorMonitor'):    
        process.l1TdeRCT.rctSourceData = 'caloStage1Digis'
        process.l1TdeRCTfromRCT.rctSourceData = 'rctDigis'
        process.l1compareforstage1.GCTsourceData = cms.InputTag("caloStage1LegacyFormatDigis")
        process.l1compareforstage1.GCTsourceEmul = cms.InputTag("valCaloStage1LegacyFormatDigis")
        process.l1compareforstage1.stage1_layer2_ = cms.bool(True)
        process.valStage1GtDigis.GctInputTag = 'caloStage1LegacyFormatDigis'

    return process
