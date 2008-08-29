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
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/18780303-826C-DD11-BB6A-000423D94E1C.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/2A192D12-826C-DD11-89D0-000423D98B5C.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/38CC78DD-816C-DD11-912A-000423D98DC4.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/42728674-826C-DD11-93A1-000423D6CA72.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/52BFF633-826C-DD11-9862-001617C3B6E8.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/5467CF6A-826C-DD11-B732-0016177CA7A0.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/5656A634-826C-DD11-A46F-001617C3B778.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/586726ED-816C-DD11-A28D-001617E30CA4.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/5ADDF21B-826C-DD11-8268-001617C3B78C.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/6666AEF8-816C-DD11-B0C3-001617DBD230.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/98E81CD6-816C-DD11-9789-000423D98844.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/9CD3FAD6-816C-DD11-A423-001617DBCF1E.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/9ECF8F16-826C-DD11-9706-000423D98EC4.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/A28BCE4E-826C-DD11-82F1-001617C3B77C.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/C42E8B3B-826C-DD11-B145-001617DBD224.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/CE71EB06-826C-DD11-A9DA-001617E30CA4.root',
'/store/relval/CMSSW_2_1_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V6_v1/0004/D28E7308-826C-DD11-8179-000423D94990.root'
#        'rfio:/castor/cern.ch/user/s/safronov/forIsoTracksFromAlCaRaw.root'
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


