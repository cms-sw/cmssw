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
    input = cms.untracked.int32(2000)
)
process.source = cms.Source("PoolSource",
    fileNames =
cms.untracked.vstring(
'/store/relval/CMSSW_3_1_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0001/0CA31FE5-CEF7-DD11-8FCD-000423D6AF24.root',
        
'/store/relval/CMSSW_3_1_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0001/1AE8394B-CEF7-DD11-AF20-000423D6006E.root',
        
'/store/relval/CMSSW_3_1_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0001/2675A3BF-06F8-DD11-9040-001617C3B710.root',
        
'/store/relval/CMSSW_3_1_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0001/428870D9-CDF7-DD11-9992-000423D99AA2.root',
        
'/store/relval/CMSSW_3_1_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0001/FA47E9E2-CEF7-DD11-811C-000423D98E6C.root'

#'file:/tmp/safronov/00E99CE2-CEF7-DD11-B46E-000423D986C4.root'
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


