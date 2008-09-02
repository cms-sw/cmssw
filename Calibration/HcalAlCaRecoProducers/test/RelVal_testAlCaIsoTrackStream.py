import FWCore.ParameterSet.Config as cms

process = cms.Process("iptRECOID2")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# put appropriate conditions tag here:
process.GlobalTag.globaltag = 'IDEAL_V6::All'

process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")

process.load("Configuration.StandardSequences.Generator_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_cff")
process.isoHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")

process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_Output_cff")

process.load("HLTrigger.Timer.timer_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)
process.source = cms.Source("PoolSource",
    fileNames =
cms.untracked.vstring(
# 2_1_4 RelVals (put your favorite):
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/00F5E713-826C-DD11-8EA1-000423D99AAE.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/0AE8CD0D-826C-DD11-9848-000423D9A212.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/0E873A03-826C-DD11-BB2B-000423D98A44.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/18780303-826C-DD11-BB6A-000423D94E1C.root'
#'file:/tmp/safronov/cvs/CMSSW_2_1_4/src/HLTrigger/Configuration/test/HLTFromDigiRaw.root'
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


