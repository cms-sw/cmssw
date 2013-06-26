import FWCore.ParameterSet.Config as cms

process = cms.Process("PAT")

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('PATLayer0Summary')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    default          = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
    PATLayer0Summary = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
        'file:/data/disk2/lowette/ttbar219.root'
     )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP_V4::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# PAT Layer 0+1
process.load("PhysicsTools.PatAlgos.patLayer0_cff")
process.load("PhysicsTools.PatAlgos.patLayer1_cff")
#process.content = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(
                process.patLayer0  
                #+ process.content # uncomment to get a dump of the output after layer 0
                + process.patLayer1  
            )

process.allLayer1Electrons.embedTrack = False
process.allLayer1Electrons.embedGsfTrack = False
process.allLayer1Electrons.embedSuperCluster = False
process.allLayer1Electrons.addResolutions = False
process.allLayer1Electrons.isolation = cms.PSet()
process.allLayer1Electrons.isoDeposits = cms.PSet()
process.allLayer1Electrons.addElectronID = False
process.allLayer1Electrons.addTrigMatch = False
process.allLayer1Electrons.addGenMatch = False
process.allLayer1Muons.embedTrack = False
process.allLayer1Muons.embedCombinedMuon = False
process.allLayer1Muons.embedStandAloneMuon = False
process.allLayer1Muons.isolation = cms.PSet()
process.allLayer1Muons.isoDeposits = cms.PSet()
process.allLayer1Muons.addResolutions = False
process.allLayer1Muons.addTrigMatch = False
process.allLayer1Muons.addGenMatch = False
process.allLayer1Jets.embedCaloTowers = False
process.allLayer1Jets.addJetCorrFactors = False
process.allLayer1Jets.addResolutions = False
process.allLayer1Jets.addBTagInfo = False
process.allLayer1Jets.addAssociatedTracks = False
process.allLayer1Jets.addJetCharge = False
process.allLayer1Jets.addTrigMatch = False
process.allLayer1Jets.addGenPartonMatch = False
process.allLayer1Jets.addGenJetMatch = False
process.allLayer1Jets.getJetMCFlavour = False
process.allLayer1Photons.embedSuperCluster = False
process.allLayer1Photons.isolation = cms.PSet()
process.allLayer1Photons.isoDeposits = cms.PSet()
process.allLayer1Photons.addPhotonID = False
process.allLayer1Photons.addTrigMatch = False
process.allLayer1Photons.addGenMatch = False
process.allLayer1Taus.embedLeadTrack = False
process.allLayer1Taus.embedSignalTracks = False
process.allLayer1Taus.embedIsolationTracks = False
process.allLayer1Taus.addResolutions = False
process.allLayer1Taus.isolation = cms.PSet()
process.allLayer1Taus.isoDeposits = cms.PSet()
process.allLayer1Taus.addTrigMatch = False
process.allLayer1Taus.addGenMatch = False

# Output module configuration
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('PATLayer1_Output.fromAOD-minimal.root'),
    # save only events passing the full path
    SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
    outputCommands = cms.untracked.vstring('drop *')
)
process.outpath = cms.EndPath(process.out)
# save PAT Layer 1 output
process.load("PhysicsTools.PatAlgos.patLayer1_EventContent_cff")
process.out.outputCommands.extend(process.patLayer1EventContent.outputCommands)

