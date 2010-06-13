import FWCore.ParameterSet.Config as cms

process = cms.Process("alcarecoHITval")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options = cms.untracked.PSet(
wantSummary=cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
'rfio:/castor/cern.ch/user/s/safronov/isoTrData/2305_ARfromRAW_0805-2205/AlCaReco_skimfromRAW_0.root',
'rfio:/castor/cern.ch/user/s/safronov/isoTrData/2305_ARfromRAW_0805-2205/AlCaReco_skimfromRAW_1.root',
'rfio:/castor/cern.ch/user/s/safronov/isoTrData/2305_ARfromRAW_0805-2205/AlCaReco_skimfromRAW_10.root',
'rfio:/castor/cern.ch/user/s/safronov/isoTrData/2305_ARfromRAW_0805-2205/AlCaReco_skimfromRAW_100.root',
'rfio:/castor/cern.ch/user/s/safronov/isoTrData/2305_ARfromRAW_0805-2205/AlCaReco_skimfromRAW_101.root',
'rfio:/castor/cern.ch/user/s/safronov/isoTrData/2305_ARfromRAW_0805-2205/AlCaReco_skimfromRAW_102.root'
))

process.load("DQMOffline.CalibCalo.MonitorHcalIsoTrackAlCaReco_cfi")
process.MonitorHcalIsoTrackAlCaReco.saveToFile=cms.bool(True)
process.MonitorHcalIsoTrackAlCaReco.outputRootFileName=cms.string("outputIsoTrackMon.root")

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.MEtoEDMConverter.verbose = cms.untracked.int32(1)

process.p = cms.Path(process.MonitorHcalIsoTrackAlCaReco)


