import FWCore.ParameterSet.Config as cms
process = cms.Process("PV")
process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.Geometry_cff") #old one, to use for old releases
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.source = cms.Source("PoolSource",
#   skipEvents=cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring(
#'root://xrootd.ba.infn.it//store/mc/RunIISpring15DR74/WprimeToWZToLLQQ_M-4000_TuneCUETP8M1_13TeV-pythia8/AODSIM/Asympt25ns_MCRUN2_74_V9-v1/10000/126A3AFB-9607-E511-A527-90B11C12EA74.root'
    'root://xrootd.ba.infn.it//store/mc/RunIISpring15DR74/WprimeToMuNu_M-5800_TuneCUETP8M1_13TeV-pythia8/AODSIM/Asympt50ns_MCRUN2_74_V9A-v2/10000/580F0D5E-760C-E511-A9D6-047D7B881D62.root'
#     'file:../../../00387B2C-4A1C-E511-A4E0-0026189438B3.root'
    )
)


process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500) )
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
#process.sortedPFPrimaryVerticesNoMET = process.sortedPFPrimaryVertices.clone(usePVMET=False)
#process.sortedPrimaryVerticesNoMET = process.sortedPrimaryVertices.clone(usePVMET=False,jets="ak4CaloJets")
#process.p = cms.Path(process.sortedPFPrimaryVertices*process.sortedPrimaryVertices*process.sortedPFPrimaryVerticesNoMET*process.sortedPrimaryVerticesNoMET)

from CommonTools.RecoAlgos.TrackWithVertexRefSelector_cfi import *
from RecoJets.JetProducers.TracksForJets_cff import *
from CommonTools.RecoAlgos.sortedPrimaryVertices_cfi import *
from RecoJets.JetProducers.caloJetsForTrk_cff import *

process.trackWithVertexRefSelectorBeforeSorting = trackWithVertexRefSelector.clone(vertexTag="offlinePrimaryVertices")
process.trackWithVertexRefSelectorBeforeSorting.ptMax=9e99
process.trackWithVertexRefSelectorBeforeSorting.ptErrorCut=9e99
process.trackRefsForJetsBeforeSorting = trackRefsForJets.clone(src="trackWithVertexRefSelectorBeforeSorting")
process.sortedPrimaryVertices.particles="trackRefsForJetsBeforeSorting"

process.p = cms.Path(process.trackWithVertexRefSelectorBeforeSorting+process.trackRefsForJetsBeforeSorting+process.sortedPrimaryVertices)

process.endpath = cms.EndPath(
    process.out
    )



