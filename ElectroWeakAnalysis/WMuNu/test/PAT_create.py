import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("ewkPAT")
process.maxEvents = cms.untracked.PSet(
      #input = cms.untracked.int32(-1)
      input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
      debugVerbosity = cms.untracked.uint32(0),
      debugFlag = cms.untracked.bool(False),
      #fileNames = cms.untracked.vstring("file:/data4/RelValWM_CMSSW_3_1_0-STARTUP31X_V1-v1_GEN-SIM-RECO/40BFAA1A-5466-DE11-B792-001D09F29533.root")
      fileNames = cms.untracked.vstring("file:/data4/Wmunu-Summer09-MC_31X_V2_preproduction_311-v1/0011/F4C91F77-766D-DE11-981F-00163E1124E7.root")
)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
            threshold = cms.untracked.string('INFO')
      ),
      destinations = cms.untracked.vstring('cout')
)
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )


# Geometry, conditions, magnetic field
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V2::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# PAT sequences
process.load("PhysicsTools.PatAlgos.patSequences_cff")
#process.content = cms.EDAnalyzer("EventContentAnalyzer")
process.pat = cms.Path( process.patDefaultSequence )

# Output: no cleaning, extra AOD collections (generator info, trigger)
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('PAT_test.root'),
    SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('pat') ),
    outputCommands = cms.untracked.vstring('drop *')
)
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoLayer1Cleaning
from PhysicsTools.PatAlgos.patEventContent_cff import patExtraAodEventContent
process.out.outputCommands.extend(patEventContentNoLayer1Cleaning)
process.out.outputCommands.extend(patExtraAodEventContent)

# End
process.outpath = cms.EndPath(process.out)
