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

    # override RAW data name to rn on data
    hltGetRaw.RawDataCollection                  = "source"
    hltGtDigis.DaqGtInputTag                     = "source"
    hltGctDigis.inputLabel                       = "source"
    hltScalersRawToDigi.scalersInputTag          = "source"
    hltSiPixelDigis.InputLabel                   = "source"
    hltMuonCSCDigis.InputObjects                 = "source"
    hltMuonDTDigis.inputLabel                    = "source"
    hltDTTFUnpacker.DTTF_FED_Source              = "source"
    hltEcalRawToRecHitFacility.sourceTag         = "source"
    hltESRawToRecHitFacility.sourceTag           = "source"
    hltHcalDigis.InputLabel                      = "source"
    hltMuonRPCDigis.InputLabel                   = "source"
    hltSiStripRawToClustersFacility.ProductLabel = "source"
    hltCscTfDigis.producer                       = "source"
    hltL1EventNumberNZS.rawInput                 = "source"
    hltDTROMonitorFilter.inputLabel              = "source"
    hltEcalCalibrationRaw.rawInputLabel          = "source"
    hltHcalCalibTypeFilter.InputTag              = "source"
    hltDTDQMEvF.inputLabel                       = "source"
    hltEcalDigis.InputLabel                      = "source"
    hltEBHltTask.FEDRawDataCollection            = "source"
    hltEEHltTask.FEDRawDataCollection            = "source"
    hltL1tfed.rawTag                             = "source"
    hltSiPixelDigisWithErrors.InputLabel         = "source"
    hltSiPixelHLTSource.RawInput                 = "source"
    hltSiStripFEDCheck.RawDataTag                = "source"

    return(process)
