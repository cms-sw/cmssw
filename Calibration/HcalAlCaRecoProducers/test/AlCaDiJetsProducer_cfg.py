import FWCore.ParameterSet.Config as cms

process = cms.Process("MYDIJETS")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(1000)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_80_120_13/GEN-SIM-RECO/MCRUN2_73_V9_71XGENSIM_FIXGT-v1/00000/8E758AAA-4DA2-E411-8068-003048FFCB96.root',
        '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_80_120_13/GEN-SIM-RECO/MCRUN2_73_V9_71XGENSIM_FIXGT-v1/00000/CE0DAE28-56A2-E411-AEFF-003048FFD79C.root',
        '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_80_120_13/GEN-SIM-RECO/MCRUN2_73_V9_71XGENSIM_FIXGT-v1/00000/D4D21D16-56A2-E411-A0C4-0026189438E2.root'
))

process.load("Calibration.HcalAlCaRecoProducers.alcadijets_cfi")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalDijets_Output_cff")

process.DiJetsRecos = cms.OutputModule("PoolOutputModule",
    outputCommands = process.OutALCARECOHcalCalDijets.outputCommands,
    fileName = cms.untracked.string('dijets.root')
)

process.p = cms.Path(process.DiJetsProd)
process.e = cms.EndPath(process.DiJetsRecos)
