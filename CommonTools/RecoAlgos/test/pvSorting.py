import FWCore.ParameterSet.Config as cms
process = cms.Process("PV")
process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.Geometry_cff") #old one, to use for old releases
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = 'POSTLS172_V4::All'


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
     '/store/relval/CMSSW_7_4_0_pre2/RelValH130GGgluonfusion_13/GEN-SIM-RECO/MCRUN2_73_V7-v1/00000/92F31767-4E85-E411-AD98-02163E00E969.root'
    )
)


process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
process.load("Configuration.EventContent.EventContent_cff")
process.out = cms.OutputModule(
    "PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('AOD.root'),
    )
process.out.outputCommands.extend(
    [
      'keep *'
    ])
process.load("CommonTools.RecoAlgos.sortedPrimaryVertices_cfi")
process.load("CommonTools.RecoAlgos.sortedPFPrimaryVertices_cfi")

process.sortedPrimaryVertices.jets = "ak4CaloJets"
process.sortedPFPrimaryVerticesNoMET = process.sortedPFPrimaryVertices.clone(usePVMET=False)
process.sortedPrimaryVerticesNoMET = process.sortedPrimaryVertices.clone(usePVMET=False,jets="ak4CaloJets")
process.p = cms.Path(process.sortedPFPrimaryVertices*process.sortedPrimaryVertices*process.sortedPFPrimaryVerticesNoMET*process.sortedPrimaryVerticesNoMET)

process.endpath = cms.EndPath(
    process.out
    )



