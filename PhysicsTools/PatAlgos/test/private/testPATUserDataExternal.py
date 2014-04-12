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
     fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/PRS/top/cmssw-data/relval200-for-pat-testing/FullSimTTBar-2_2_X_2008-11-03-STARTUP_V7-AODSIM.100.root')
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP_V4::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# PAT Layer 0+1
process.load("PhysicsTools.PatAlgos.patLayer0_cff")
process.load("PhysicsTools.PatAlgos.patLayer1_cff")

process.answer = cms.EDProducer("PATUserDataTestModule", # each of this will produce all
    mode  = cms.string("external"),                      # but I don't care
    muons = cms.InputTag("allLayer0Muons"),
    label = cms.string("il"),                            # use instance label 'il' for the ints
)
process.pi = cms.EDProducer("PATUserDataTestModule",
    mode  = cms.string("external"),
    muons = cms.InputTag("allLayer0Muons"),
)
process.halfP4 = cms.EDProducer("PATUserDataTestModule",
    mode  = cms.string("external"),
    muons = cms.InputTag("allLayer0Muons"),
)
process.allLayer1Muons.userData.userInts.src    = cms.VInputTag(cms.InputTag("answer","il"))
process.allLayer1Muons.userData.userFloats.src  = cms.VInputTag(cms.InputTag("pi"))
process.allLayer1Muons.userData.userClasses.src = cms.VInputTag(cms.InputTag("halfP4"))
process.testRead = cms.EDProducer("PATUserDataTestModule",
    mode  = cms.string("read"),
    muons = cms.InputTag("selectedLayer1Muons"),
)



process.content = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(
                process.patLayer0  
                + process.answer * process.pi * process.halfP4
                #+ process.content # uncomment to get a dump of the output after layer 0
                + process.patLayer1  
                + process.testRead
            )

# Output module configuration
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('PATLayer1_Output.fromAOD_full.root'),
    # save only events passing the full path
    SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
    outputCommands = cms.untracked.vstring('drop *')
)
process.outpath = cms.EndPath(process.out)
# save PAT Layer 1 output
process.load("PhysicsTools.PatAlgos.patLayer1_EventContent_cff")
process.out.outputCommands.extend(process.patLayer1EventContent.outputCommands)

