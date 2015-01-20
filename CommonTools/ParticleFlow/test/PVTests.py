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
    'file:many/ZH_HToBB_ZToNuNu_M-125_13TeV_powheg-herwigpp-PU40bx25_PHYS14_25_V1-v1-40778748-2072-E411-8840-00266CFEFE1C.root'
#/store/relval/CMSSW_7_4_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU50ns_MCRUN2_73_V6-v1/00000/14407C47-BC9A-E411-84F3-002590596498.root'
    )
)


process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20) )
process.load("Configuration.EventContent.EventContent_cff")
process.out = cms.OutputModule(
    "PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('AOD.root'),
    )

process.load("CommonTools.ParticleFlow.gedPrimaryVertices_cfi")
process.out.outputCommands.extend(
    [
      'keep *'
    ])
process.gedPrimaryVerticesNoMET = process.gedPrimaryVertices.clone(usePVMET=False)
process.tkPrimaryVerticesNoMET = process.tkPrimaryVertices.clone(usePVMET=False)
process.p = cms.Path(process.gedPrimaryVertices*process.tkPrimaryVertices*process.gedPrimaryVerticesNoMET*process.tkPrimaryVerticesNoMET)

process.endpath = cms.EndPath(
    process.out
    )



