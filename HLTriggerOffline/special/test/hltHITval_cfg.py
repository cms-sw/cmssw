import FWCore.ParameterSet.Config as cms

process = cms.Process("HITval")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # put here 10 TeV RelVal minbias sample with HLTDEBUG, RAW & RECO
    fileNames = cms.untracked.vstring(
#'file:8E29_HLT_900GeV_MC_startupBS_4tim.root'
#'file:8E29_HLT_fDataTest_skimPDLS_4tim_bare336.root'
)
)
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR09_P_V8::All'

process.load("HLTrigger.HLTfilters.hltHighLevel_cfi")
process.hltHighLevel.HLTPaths = ['HLT_IsoTrackHB_8E29','HLT_IsoTrackHE_8E29']

process.load("HLTriggerOffline.special.hltHITval_cfi")

process.hltHITval.L1FilterName=cms.string("hltL1sIsoTrack8E29")
process.hltHITval.L2producerLabelHB=cms.InputTag("hltIsolPixelTrackProdHB8E29")
process.hltHITval.L2producerLabelHE=cms.InputTag("hltIsolPixelTrackProdHE8E29")
process.hltHITval.hltL3FilterLabelHB=cms.string("hltIsolPixelTrackL3FilterHB8E29")
process.hltHITval.hltL3FilterLabelHE=cms.string("hltIsolPixelTrackL3FilterHE8E29")
process.hltHITval.L3producerLabelHB=cms.InputTag("hltHITIPTCorrectorHB8E29")
process.hltHITval.L3producerLabelHE=cms.InputTag("hltHITIPTCorrectorHE8E29")
process.hltHITval.hltProcessName=cms.string("HLT2")
process.hltHITval.checkL1=cms.bool(False)
process.hltHITval.doL1Prescaling=cms.bool(False)
process.hltHITval.produceRates=cms.bool(True)
process.hltHITval.luminosity=cms.double(8e29)
process.hltHITval.sampleCrossSection=cms.double(7.53E10)
process.hltHITval.saveToRootFile=cms.bool(True)
process.hltHITval.HBtriggerName=cms.string("HLT_IsoTrackHB_8E29")
process.hltHITval.HEtriggerName=cms.string("HLT_IsoTrackHE_8E29")

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.MEtoEDMConverter.verbose = cms.untracked.int32(1)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.dqmOut = cms.OutputModule("PoolOutputModule",
     fileName = cms.untracked.string('dqmHltHITval_minBias.root'),
     outputCommands = cms.untracked.vstring("drop *", "keep *_MEtoEDMConverter_*_*")
 )

process.p = cms.Path(process.hltHITval)

#process.ep=cms.EndPath(process.dqmOut)
