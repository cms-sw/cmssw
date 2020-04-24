import FWCore.ParameterSet.Config as cms

process = cms.Process("PHISYM")


process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = 
#cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0001/087DC4B2-640A-DE11-86E5-000423D98DD4.root')
cms.untracked.vstring('file:/afs/cern.ch/cms/Tutorials/TWIKI_DATA/CMSDataAnaSch_RelValZMM536.root')
)

process.PhiSymIter = cms.EDProducer("AlCaEcalHcalReadoutsProducer",
    hbheInput = cms.InputTag("hbhereco"),
    hoInput = cms.InputTag("horeco"),
    hfInput = cms.InputTag("hfreco")
)

process.PhiSymOut = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *',
                                            "keep *_horeco_*_*", 
                                            "keep *_hfreco_*_*",
                                            "keep *_hbhereco_*_*",
                                            "keep *_offlinePrimaryVertices_*_*",
                                            "keep edmTriggerResults_*_*_HLT"
),
    fileName = cms.untracked.string('phi_sym.root')
)

process.p = cms.Path(process.PhiSymIter)
process.e = cms.EndPath(process.PhiSymOut)

