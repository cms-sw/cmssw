import FWCore.ParameterSet.Config as cms


process = cms.Process("validation")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("DQMServices.Core.DQM_cfg")

# load the full reconstraction configuration, to make sure we're getting all needed dependencies
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")


process.load('RecoVertex/AdaptiveVertexFinder/inclusiveVertexing_cff')
process.load('RecoBTag/SecondaryVertex/bToCharmDecayVertexMerger_cfi')




process.IVFAnalyzer = cms.EDAnalyzer('IVFAnalyzer')


process.plots = cms.Path(process.inclusiveVertexing * process.inclusiveMergedVerticesFiltered * process.bToCharmDecayVertexMerged * process.IVFAnalyzer )



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.PoolSource.fileNames = [
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/F21CBC5F-5790-E111-982A-0018F3D096F6.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/7C9DC3A3-8890-E111-987E-00261894393C.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/FC83FEFC-6090-E111-81A8-0026189438A9.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/0E73FD2F-9C90-E111-BF5F-002618943954.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/B64440BF-8C90-E111-9C3A-00261894389E.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/5AB9E8F0-7B90-E111-AD3D-0026189438A2.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/A68391A4-8B90-E111-8430-003048D42DC8.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/8EF7DA70-8D90-E111-A4FE-0026189438AD.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/786CC344-4F90-E111-967F-00261894383B.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/648A880D-9E90-E111-9D4B-003048FFCC2C.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/40E0AAC0-9190-E111-B3FF-002618943908.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/0C61F494-5E90-E111-A011-0026189438C9.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/8CFEFC53-6F90-E111-90E4-002618943821.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/D0718FB3-9390-E111-9E79-003048678DA2.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/7CD6D78E-9B90-E111-9C5E-0018F3D09700.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/D0D85C7F-9890-E111-9359-0018F3D09680.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/D6E10BC1-9390-E111-967D-002618943876.root',
'/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/A4A71598-8390-E111-A375-00304867924A.root'
]


# global tag
process.GlobalTag.globaltag = cms.string( "START52_V9::All")
