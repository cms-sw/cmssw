import FWCore.ParameterSet.Config as cms

process = cms.Process("alcarecoHITval")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
'rfio:/castor/cern.ch/user/s/safronov/testDQM/rawToReco_IsoTr_HLT_IDEAL_var1.root'
))

process.load("DQMOffline.CalibCalo.MonitorHcalIsoTrackAlCaReco_cfi")
process.MonitorHcalIsoTrackAlCaReco.hltL3FilterLabel=cms.InputTag('hltHITCorTracksFilter1::HLT1')
process.MonitorHcalIsoTrackAlCaReco.outputRootFileName=cms.string("MonitorHcalIsoTrackAlCaReco.root")

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.MEtoEDMConverter.verbose = cms.untracked.int32(1)

process.dqmOut = cms.OutputModule("PoolOutputModule",
     fileName = cms.untracked.string('dqmAlCaRecoHITval_IDEAL.root'),
     outputCommands = cms.untracked.vstring("drop *", "keep *_MEtoEDMConverter_*_*")
 )

process.p = cms.Path(process.MonitorHcalIsoTrackAlCaReco + process.MEtoEDMConverter)

process.ep=cms.EndPath(process.dqmOut)

