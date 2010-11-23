import FWCore.ParameterSet.Config as cms

from L1Trigger.Configuration import patchToRerunL1Emulator


def Base(process):
#   default modifications

    process.options.wantSummary = cms.untracked.bool(True)

    process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    process.MessageLogger.categories.append('L1GtTrigReport')
    process.MessageLogger.categories.append('HLTrigReport')

    if 'hltTrigReport' in process.__dict__:
        process.hltTrigReport.HLTriggerResults = cms.InputTag( 'TriggerResults','',process.name_() )

    if 'hltDQMHLTScalers' in process.__dict__:
        process.hltDQMHLTScalers.triggerResults = cms.InputTag( 'TriggerResults','',process.name_() )

    if 'hltDQML1SeedLogicScalers' in process.__dict__:
        process.hltDQML1SeedLogicScalers.processname = process.name_()

    return(process)


def L1T(process):
#   modifications when running L1T only

    process.load('L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi')
    process.l1GtTrigReport.L1GtRecordInputTag = cms.InputTag( "simGtDigis" )

    process.L1AnalyzerEndpath = cms.EndPath( process.l1GtTrigReport )
    process.schedule.append(process.L1AnalyzerEndpath)

    process=Base(process)

    return(process)


def L1THLT(process):
#   modifications when running L1T+HLT

    process=Base(process)

    return(process)


def L1THLT2(process):
#   modifications when re-running L1T+HLT    

#   run trigger primitive generation on unpacked digis, then central L1

    process.load("L1Trigger.Configuration.CaloTriggerPrimitives_cff")
    process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
    process.simHcalTriggerPrimitiveDigis.inputLabel = ('hcalDigis', 'hcalDigis')

#   patch the process to use 'sim*Digis' from the L1 emulator
#   instead of 'hlt*Digis' from the RAW data

    patchToRerunL1Emulator.switchToSimGtDigis( process )

    process=Base(process)

    return(process)


def HLTData(process):
#   modifications when running on real data (currently pp [not HI] only!)

#   drop on input the previous HLT results
    process.source.inputCommands = cms.untracked.vstring (
        'keep *',
        'drop *_hltL1GtObjectMap_*_*',
        'drop *_TriggerResults_*_*',
        'drop *_hltTriggerSummaryAOD_*_*',
    )

#   override the L1 menu
    if 'toGet' not in process.GlobalTag.__dict__:
        process.GlobalTag.toGet = cms.VPSet()
    process.GlobalTag.toGet.append(
        cms.PSet(  
            record  = cms.string( "L1GtTriggerMenuRcd" ),
            tag     = cms.string( "L1GtTriggerMenu_L1Menu_Commissioning2010_v4_mc" ),
            connect = cms.untracked.string( process.GlobalTag.connect.value().replace('CMS_COND_31X_GLOBALTAG', 'CMS_COND_31X_L1T') )
        )
    )

#   override RAW data name to rn on data
    process.hltFEDSelector.inputTag                      = "source"
    process.hltGetRaw.RawDataCollection                  = "source"
    process.hltGtDigis.DaqGtInputTag                     = "source"
    process.hltGctDigis.inputLabel                       = "source"
    process.hltScalersRawToDigi.scalersInputTag          = "source"
    process.hltSiPixelDigis.InputLabel                   = "source"
    process.hltMuonCSCDigis.InputObjects                 = "source"
    process.hltMuonDTDigis.inputLabel                    = "source"
    process.hltDTTFUnpacker.DTTF_FED_Source              = "source"
    process.hltEcalRawToRecHitFacility.sourceTag         = "source"
    process.hltESRawToRecHitFacility.sourceTag           = "source"
    process.hltHcalDigis.InputLabel                      = "source"
    process.hltMuonRPCDigis.InputLabel                   = "source"
    process.hltSiStripRawToClustersFacility.ProductLabel = "source"
    process.hltL1EventNumberNZS.rawInput                 = "source"
    process.hltDTROMonitorFilter.inputLabel              = "source"
    process.hltEcalCalibrationRaw.inputTag               = "source"
    process.hltHcalCalibTypeFilter.InputTag              = "source"
    process.hltDTDQMEvF.inputLabel                       = "source"
    process.hltEcalDigis.InputLabel                      = "source"
    process.hltEBHltTask.FEDRawDataCollection            = "source"
    process.hltEEHltTask.FEDRawDataCollection            = "source"
    process.hltL1tfed.rawTag                             = "source"
    process.hltSiPixelDigisWithErrors.InputLabel         = "source"
    process.hltSiPixelHLTSource.RawInput                 = "source"
    process.hltSiStripFEDCheck.RawDataTag                = "source"

    process=Base(process)
    
    return(process)
