import FWCore.ParameterSet.Config as cms

process = cms.Process("alcarecoHITval")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
'file:/tmp/sergeant/ALCARECOHcalCalIsoTrk_validation2.root'
))

process.MonitorHcalIsoTrackAlCaReco = cms.EDAnalyzer("ValidationHcalIsoTrackAlCaReco",
folderName=cms.string("Calibration/HcalCalibAlgos/plugins"),
saveToFile=cms.bool(True),
outputRootFileName=cms.string("HcalIsoTrackAlCaRecoMon.root"),
hltTriggerEventLabel=cms.InputTag('hltTriggerSummaryAOD'),
hltL3FilterLabel=cms.InputTag('hltIsolPixelTrackFilter::HLT'),
alcarecoIsoTracksLabel=cms.InputTag('IsoProd:HcalIsolatedTrackCollection'),
recoTracksLabel=cms.InputTag('IsoProd:IsoTrackTracksCollection'),
simTracksTag = cms.InputTag('g4SimHits')
)

#process.load("Calibration.HcalCalibAlgos.MonitorHcalIsoTrackAlCaReco_cfi")
#process.load("DQMOffline.CalibCalo.PostProcessorHcalIsoTrack_cfi")
#process.MonitorHcalIsoTrackAlCaReco.hltL3FilterLabel=cms.InputTag('hltHITCorTracksFilter1::HLT1')
process.MonitorHcalIsoTrackAlCaReco.outputRootFileName=cms.string("MonitorHcalIsoTrackAlCaReco.root")

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.MEtoEDMConverter.verbose = cms.untracked.int32(1)

process.dqmOut = cms.OutputModule("PoolOutputModule",
     fileName = cms.untracked.string('dqmAlCaRecoHITval_IDEAL.root'),
     outputCommands = cms.untracked.vstring("drop *", "keep *_MEtoEDMConverter_*_*")
 )

process.p = cms.Path(process.MonitorHcalIsoTrackAlCaReco + process.MEtoEDMConverter)

process.ep=cms.EndPath(process.dqmOut)

