import FWCore.ParameterSet.Config as cms

process = cms.Process("MYITERATIVEPHISYM")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond_condDBv2 import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(100)
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = 
cms.untracked.vstring(
#   'file:/tmp/andriusj/6EC8FCC8-E2A8-E411-9506-002590596468.root'
       '/store/relval/CMSSW_7_4_0_pre6/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/6EC8FCC8-E2A8-E411-9506-002590596468.root'
#       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_80_120_13/GEN-SIM-RECO/MCRUN2_73_V9_71XGENSIM_FIXGT-v1/00000/D4D21D16-56A2-E411-A0C4-0026189438E2.root'
#       '/store/relval/CMSSW_7_3_0/RelValQCD_Pt_80_120_13/GEN-SIM-RECO/MCRUN2_73_V9_71XGENSIM_FIXGT-v1/00000/8E758AAA-4DA2-E411-8068-003048FFCB96.root'
 )
)

process.load("Calibration.HcalAlCaRecoProducers.alcaiterphisym_cfi")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIterativePhiSym_Output_cff")


process.IterativePhiSym = cms.OutputModule("PoolOutputModule",
   outputCommands = process.OutALCARECOHcalCalIterativePhiSym.outputCommands,
   fileName = cms.untracked.string('iter_phi_sym.root')
)

process.p = cms.Path(process.IterativePhiSymProd)
process.e = cms.EndPath(process.IterativePhiSym)
