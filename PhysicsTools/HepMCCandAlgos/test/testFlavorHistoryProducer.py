# Import configurations
import FWCore.ParameterSet.Config as cms


process = cms.Process("testFlavorHistoryProducer")

        
# set the number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)




# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.PATLayer0Summary=dict()
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    default          = cms.untracked.PSet( limit = cms.untracked.int32(0)  )
)
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Load geometry
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('IDEAL_V9::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# input MC stuff
process.load( "SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load( "PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.load( "PhysicsTools.HepMCCandAlgos.genEventWeight_cfi")
process.load( "PhysicsTools.HepMCCandAlgos.genEventScale_cfi")

process.load( "RecoJets.Configuration.GenJetParticles_cff")
process.load( "RecoJets.JetProducers.SISConeJetParameters_cfi" )
process.load( "RecoJets.JetProducers.GenJetParameters_cfi" )
process.load( "RecoJets.JetProducers.FastjetParameters_cfi" )
process.load( "RecoJets.JetProducers.sisCone5GenJets_cff")


# input flavor history stuff
process.load("PhysicsTools.HepMCCandAlgos.flavorHistoryPaths_cfi")

process.printList = cms.EDAnalyzer( "ParticleListDrawer",
                                    src =  cms.InputTag( "genParticles" ),
                                    maxEventsToPrint = cms.untracked.int32( 10 )
#                                    printOnlyHardInteraction = cms.untracked.bool( True )
)




# request a summary at the end of the file
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# define the source, from reco input

process.source = cms.Source("PoolSource",
                        debugVerbosity = cms.untracked.uint32(200),
                        debugFlag = cms.untracked.bool(True),
                        
                        fileNames = cms.untracked.vstring(
        '/store/mc/Summer08/WJets-madgraph/USER/IDEAL_V9_PAT_v4/0015/D8AE36D9-F6EC-DD11-B688-001A92971AA8.root'
        )
                        )


# define event selection to be that which satisfies 'p'
process.eventSel = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)



# load the different paths to make the different HF selections


import PhysicsTools.HepMCCandAlgos.flavorHistoryPaths_cfi as flavortools


process.p         = cms.Path( flavortools.flavorHistorySeq )


# Set the threshold for output logging to 'info'
process.MessageLogger.cerr.threshold = 'INFO'


# talk to output module


process.out = cms.OutputModule( "PoolOutputModule",
  process.eventSel,
  fileName = cms.untracked.string( "testFlavorHistoryProducer.root" ),
  outputCommands= cms.untracked.vstring(
    "drop *",
    "keep *_sisCone5GenJets_*_*",
    "keep *_genParticles_*_*",
    "keep *_genEventWeight_*_*",
    "keep *_bFlavorHistoryProducer_*_*",
    "keep *_cFlavorHistoryProducer_*_*",
    "keep *_flavorHistoryFilter_*_*"
    )
                                )


# define output path
process.outpath = cms.EndPath(process.out)
