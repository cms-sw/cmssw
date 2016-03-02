import FWCore.ParameterSet.Config as cms

import os

##############################################################################
# customisations for L1T demos
#
# Add demonstration modules to cmsDriver customs.
#
##############################################################################

def L1TBasicDemo(process):
    print "L1T INFO:  adding basic demo module to the process."
    process.load('L1Trigger.L1TCommon.l1tBasicDemo_cfi')
    process.l1tBasicDemoPath = cms.Path(process.l1tBasicDemo)
    process.schedule.append(process.l1tBasicDemoPath)
    return process

def L1THLTDemo(process):
    print "L1T INFO:  adding HLT demo module to the process."
    #
    # BEGIN HLT UNPACKER SEQUENCE FOR STAGE 2
    #

    process.hltGtStage2Digis = cms.EDProducer(
        "L1TRawToDigi",
        Setup           = cms.string("stage2::GTSetup"),
        FedIds          = cms.vint32( 1404 ),
        )
    
    process.hltCaloStage2Digis = cms.EDProducer(
        "L1TRawToDigi",
        Setup           = cms.string("stage2::CaloSetup"),
        FedIds          = cms.vint32( 1360, 1366 ),
        )
    
    process.hltGmtStage2Digis = cms.EDProducer(
        "L1TRawToDigi",
        Setup = cms.string("stage2::GMTSetup"),
        FedIds = cms.vint32(1402),
        )
    
    process.hltGtStage2ObjectMap = cms.EDProducer(
        "L1TGlobalProducer",
        GmtInputTag = cms.InputTag("hltGmtStage2Digis"),
        ExtInputTag = cms.InputTag("hltGtStage2Digis"), # (external conditions are not emulated, use unpacked)
        CaloInputTag = cms.InputTag("hltCaloStage2Digis"),
        AlgorithmTriggersUnprescaled = cms.bool(True),
        AlgorithmTriggersUnmasked = cms.bool(True),
        )

    # keeping same sequence name as for legacy system:
    process.HLTL1UnpackerSequence = cms.Sequence(
        process.hltGtStage2Digis +
        process.hltCaloStage2Digis +
        process.hltGmtStage2Digis +
        process.hltGtStage2ObjectMap
        )

    #
    # END HLT UNPACKER SEQUENCE FOR STAGE 2
    #

    #
    # BEGIN L1T SEEDS EXAMPLE FOR STAGE 2
    #
    process.hltL1TSeed = cms.EDFilter( 
        "HLTL1TSeed",
        L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet36 AND L1_SingleEG10" ),
        SaveTags             = cms.bool( True ),
        L1ObjectMapInputTag  = cms.InputTag("hltGtStage2ObjectMap"),
        L1GlobalInputTag     = cms.InputTag("hltGtStage2Digis"),
        L1MuonInputTag       = cms.InputTag("hltGmtStage2Digis"),
        L1EGammaInputTag     = cms.InputTag("hltCaloStage2Digis"),
        L1JetInputTag        = cms.InputTag("hltCaloStage2Digis"),
        L1TauInputTag        = cms.InputTag("hltCaloStage2Digis"),
        L1EtSumInputTag      = cms.InputTag("hltCaloStage2Digis"),
        )

    # HLT Seed sequence
    process.HLTL1TSeedSequence  = cms.Sequence( 
        process.hltL1TSeed 
        )

    #
    # END L1T SEEDS EXAMPLE FOR STAGE 2
    #

    print "L1T INFO:  will dump a summary of Stage2 content as unpacked by HLT to screen."    
    process.load('L1Trigger.L1TCommon.l1tSummaryStage2HltDigis_cfi')

    # gt analyzer
    process.l1tGlobalSummary = cms.EDAnalyzer(
        'L1TGlobalSummary',
        AlgInputTag = cms.InputTag("hltGtStage2ObjectMap"),
        ExtInputTag = cms.InputTag("hltGtStage2ObjectMap"),
        # DumpTrigResults = cms.bool(True), # per event dump of trig results
        DumpTrigSummary = cms.bool(True), # job summary... not yet implemented...
        )

    process.HLTL1TDebugSequence  = cms.Sequence(process.l1tSummaryStage2HltDigis + process.l1tGlobalSummary)



    print "L1T Input:  HLTL1UnpackerSequence:  "
    print process.HLTL1UnpackerSequence
    print "L1T Input:  HLTL1TSeedSequence:  "
    print process.HLTL1TSeedSequence
    print "L1T Input:  HLTL1TDebugSequence:  "
    print process.HLTL1TDebugSequence
    process.l1tHLTDemoPath = cms.Path(process.HLTL1UnpackerSequence + process.HLTL1TSeedSequence + process.HLTL1TDebugSequence)
    process.schedule.append(process.l1tHLTDemoPath)
    return process
