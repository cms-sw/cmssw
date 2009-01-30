import FWCore.ParameterSet.Config as cms

process = cms.Process("Rec")

process.load('Configuration/EventContent/EventContent_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/2008/7/15/RelVal-RelValMinBias-STARTUP_V4_InitialLumiPileUp_v1-2nd/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre8-RelVal-STARTUP_V4_InitialLumiPileUp_v1-2nd-STARTUP_V4-unmerged/0001/44ED7716-9B52-DD11-A23E-0019DB29C614.root',
       '/store/relval/2008/7/15/RelVal-RelValMinBias-STARTUP_V4_InitialLumiPileUp_v1-2nd/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre8-RelVal-STARTUP_V4_InitialLumiPileUp_v1-2nd-STARTUP_V4-unmerged/0001/5A26A125-9B52-DD11-9111-00161757BF42.root',
       '/store/relval/2008/7/15/RelVal-RelValMinBias-STARTUP_V4_InitialLumiPileUp_v1-2nd/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre8-RelVal-STARTUP_V4_InitialLumiPileUp_v1-2nd-STARTUP_V4-unmerged/0001/820AE631-9D52-DD11-AEB3-0019DB29C620.root',
       '/store/relval/2008/7/15/RelVal-RelValMinBias-STARTUP_V4_InitialLumiPileUp_v1-2nd/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre8-RelVal-STARTUP_V4_InitialLumiPileUp_v1-2nd-STARTUP_V4-unmerged/0001/84CFE054-9A52-DD11-B1AA-001617C3B70E.root',
       '/store/relval/2008/7/15/RelVal-RelValMinBias-STARTUP_V4_InitialLumiPileUp_v1-2nd/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre8-RelVal-STARTUP_V4_InitialLumiPileUp_v1-2nd-STARTUP_V4-unmerged/0001/98023027-9B52-DD11-9936-0016177CA778.root',
       '/store/relval/2008/7/15/RelVal-RelValMinBias-STARTUP_V4_InitialLumiPileUp_v1-2nd/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre8-RelVal-STARTUP_V4_InitialLumiPileUp_v1-2nd-STARTUP_V4-unmerged/0001/CA3D9B59-9A52-DD11-9D7A-001617C3B5D8.root',
       '/store/relval/2008/7/15/RelVal-RelValMinBias-STARTUP_V4_InitialLumiPileUp_v1-2nd/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre8-RelVal-STARTUP_V4_InitialLumiPileUp_v1-2nd-STARTUP_V4-unmerged/0001/CE71E5E0-9A52-DD11-B509-001617C3B6DE.root')
)



process.bscTrigger=cms.EDProducer("BSCTrigger",
                          bitNumbers=cms.vuint32(36,37,38,39,40,41),
                          bitPrescales=cms.vuint32(1,1,1,1,1,1),
                          bitNames=cms.vstring('BSC_H_IP','BSC_H_IM','BSC_H_OP','BSC_H_OM','BSC_MB_I','BSC_MB_O')
                          )

process.MessageLogger = cms.Service("MessageLogger",
    suppressInfo = cms.untracked.vstring(),
    debugs = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    suppressDebug = cms.untracked.vstring(),
    cout = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    warnings = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    default = cms.untracked.PSet(

    ),
    errors = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    cerr = cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(False),
        BscSim = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        FwkReport = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkSummary = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('DEBUG')
    ),
    FrameworkJobReport = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    suppressWarning = cms.untracked.vstring(),
    statistics = cms.untracked.vstring('cerr_stats'),
    cerr_stats = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        output = cms.untracked.string('cerr')
    ),
    infos = cms.untracked.PSet(
        Root_NoDictionary = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        placeholder = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('warnings', 
        'errors', 
        'infos', 
        'debugs', 
        'cout', 
        'cerr'),
    debugModules = cms.untracked.vstring('bscTrigger'),
    categories = cms.untracked.vstring('BscSim','FwkJob', 
        'FwkReport', 
        'FwkSummary', 
        'Root_NoDictionary'),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport')
)
# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)




# Event output

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('/tmp/BSCTrigger.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('trigger_step')
    )
)

# Path and EndPath definitions

process.trigger_step = cms.Path(process.bscTrigger)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.trigger_step,process.outpath)




