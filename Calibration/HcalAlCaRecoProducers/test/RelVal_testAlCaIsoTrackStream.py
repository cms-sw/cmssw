import FWCore.ParameterSet.Config as cms

process = cms.Process("iptRECOID2")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# put appropriate conditions tag here:
process.GlobalTag.globaltag = 'STARTUP_30X::All'

process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")

process.load("Configuration.StandardSequences.Generator_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_cff")
process.isoHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
process.IsoProd.hltL3FilterLabel = cms.InputTag("hltIsolPixelTrackFilter::HLT")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_Output_cff")

process.load("HLTrigger.Timer.timer_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
    fileNames =
cms.untracked.vstring(
# 3_0_0_pre7 RelVals (put your favorite):
'/store/relval/CMSSW_3_0_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/0237F671-86E9-DD11-A724-003048678B06.root',
'/store/relval/CMSSW_3_0_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/0C7C2ABA-88E9-DD11-94DF-001A9281171C.root',
'/store/relval/CMSSW_3_0_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/22619C7E-93E9-DD11-AB48-003048767ECB.root',
'/store/relval/CMSSW_3_0_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/226EDFF8-91E9-DD11-A2B3-001A928116EA.root',
'/store/relval/CMSSW_3_0_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/2489A87D-86E9-DD11-907A-001731AF6A85.root',
'/store/relval/CMSSW_3_0_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/2A52D9C4-85E9-DD11-B665-0030486790FE.root',
'/store/relval/CMSSW_3_0_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/4AF58617-95E9-DD11-9497-003048D15CC0.root',
'/store/relval/CMSSW_3_0_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/4CC3DA6B-86E9-DD11-A91A-00304867905A.root',
'/store/relval/CMSSW_3_0_0_pre7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0002/5815387E-86E9-DD11-A45F-001731AF68C3.root'
)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.TimerService = cms.Service("TimerService",
    useCPUtime = cms.untracked.bool(True)
)

process.pts = cms.EDFilter("PathTimerInserter")

process.PathTimerService = cms.Service("PathTimerService")

process.hltPoolOutput = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('rawToReco_IsoTr_HLT_TEST.root'),
    outputCommands = cms.untracked.vstring('keep *_IsoProd_*_*')
)

process.AlCaIsoTrTest = cms.Path(process.seqALCARECOHcalCalIsoTrk)
process.HLTPoolOutput = cms.EndPath(process.pts*process.hltPoolOutput)


