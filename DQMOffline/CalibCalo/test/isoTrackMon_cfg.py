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
# Output definition

process.DQMOutput = cms.OutputModule("PoolOutputModule",
                                     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
                                     fileName = cms.untracked.string('dqm.root'),
)


process.load("DQMOffline.CalibCalo.MonitorHcalIsoTrackAlCaReco_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.MEtoEDMConverter.verbose = cms.untracked.int32(1)

process.p1 = cms.Path(process.MonitorHcalIsoTrackAlCaReco)
process.p2 = cms.Path(process.MEtoEDMConverter)
process.op = cms.EndPath(process.DQMOutput)

# Schedule definition
process.schedule = cms.Schedule(process.p1,process.p2,process.op)
