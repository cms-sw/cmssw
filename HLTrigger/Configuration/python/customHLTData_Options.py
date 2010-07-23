import FWCore.ParameterSet.Config as cms
def customise(process):
    if 'hltTrigReport' in process.__dict__:
        process.hltTrigReport.HLTriggerResults = cms.InputTag( 'TriggerResults','',process.name_() )

    if 'hltDQMHLTScalers' in process.__dict__:
        process.hltDQMHLTScalers.triggerResults = cms.InputTag( 'TriggerResults','',process.name_() )

    if 'hltDQML1SeedLogicScalers' in process.__dict__:
        process.hltDQML1SeedLogicScalers.processname = process.name_()

    process.options.wantSummary = cms.untracked.bool(True)
    process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    process.MessageLogger.categories.append('L1GtTrigReport')
    process.MessageLogger.categories.append('HLTrigReport')

    # drop on input the previous HLT results
    #process.source.inputCommands = cms.untracked.vstring (
    #    'keep *',
    #    'drop *_hltL1GtObjectMap_*_*',
    #    'drop *_TriggerResults_*_*',
    #    'drop *_hltTriggerSummaryAOD_*_*',
    #)

    # override RAW data name to rn on data
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
    process.hltCscTfDigis.producer                       = "source"
    process.hltL1EventNumberNZS.rawInput                 = "source"
    process.hltDTROMonitorFilter.inputLabel              = "source"
    process.hltEcalCalibrationRaw.rawInputLabel          = "source"
    process.hltHcalCalibTypeFilter.InputTag              = "source"
    process.hltDTDQMEvF.inputLabel                       = "source"
    process.hltEcalDigis.InputLabel                      = "source"
    process.hltEBHltTask.FEDRawDataCollection            = "source"
    process.hltEEHltTask.FEDRawDataCollection            = "source"
    process.hltL1tfed.rawTag                             = "source"
    process.hltSiPixelDigisWithErrors.InputLabel         = "source"
    process.hltSiPixelHLTSource.RawInput                 = "source"
    process.hltSiStripFEDCheck.RawDataTag                = "source"

    return(process)
