import FWCore.ParameterSet.Config as cms

process = cms.Process("REPROD")





# mm
# mm
# mm

# GEN-REC muon matching
process.genMatchMap = cms.EDFilter("MCTruthDeltaRMatcherNew",
    src = cms.InputTag("muons"),
    matched = cms.InputTag("genParticles"),
    distMin = cms.double(0.15),
    matchPDGId = cms.vint32(13)
)





# Create a new "distorted" PFCandidate collection
process.distortedPFCand = cms.EDFilter("DistortedPFCandProducer",
      MuonTag = cms.untracked.InputTag("muons"),
      PFTag = cms.untracked.InputTag("particleFlow"),
      GenMatchTag = cms.untracked.InputTag("genMatchMap"),

      EtaBinEdges = cms.untracked.vdouble(-3.,3.), # one more entry than next vectors

      ShiftOnOneOverPt = cms.untracked.vdouble(1.e-3), #in [1/GeV] units
      RelativeShiftOnPt = cms.untracked.vdouble(0.), # relative
      UncertaintyOnOneOverPt = cms.untracked.vdouble(3.e-3), #in [1/GeV] units
      RelativeUncertaintyOnPt = cms.untracked.vdouble(3.e-3), # relative

      EfficiencyRatioOverMC = cms.untracked.vdouble(1.)
)



# Create the old distortedMuon collection
process.distortedMuons = cms.EDFilter("DistortedMuonProducer",
      MuonTag = cms.untracked.InputTag("muons"),
      GenMatchTag = cms.untracked.InputTag("genMatchMap"),

      EtaBinEdges = cms.untracked.vdouble(-3.,3.), # one more entry than next vectors

      ShiftOnOneOverPt = cms.untracked.vdouble(1.e-3), #in [1/GeV] units
      RelativeShiftOnPt = cms.untracked.vdouble(0.), # relative
      UncertaintyOnOneOverPt = cms.untracked.vdouble(3.e-3), #in [1/GeV] units
      RelativeUncertaintyOnPt = cms.untracked.vdouble(3.e-3), # relative

      EfficiencyRatioOverMC = cms.untracked.vdouble(1.)
)

 




# mm 
# mm
# mm



process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START3X_V26A::All'
#process.load("Configuration.StandardSequences.FakeConditions_cff")

# process.Timing =cms.Service("Timing")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
#        'file:/data4/Wmunu_Summer09-MC_31X_V3-v1_GEN-SIM-RECO/0009/76E35258-507F-DE11-9A21-0022192311C5.root'
          '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEFC70B6-F53D-DF11-B57E-003048679150.root'
    ),
    secondaryFileNames = cms.untracked.vstring(),
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

#process.MessageLogger = cms.Service("MessageLogger",
#    rectoblk = cms.untracked.PSet(
#        threshold = cms.untracked.string('INFO')
#    ),
#    destinations = cms.untracked.vstring('rectoblk')
#)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.load("ElectroWeakAnalysis.Utilities.distPFMET_cfi") 


process.p1 = cms.Path(process.genMatchMap+process.distortedMuons
             +process.distortedPFCand+process.distpfMet)


# And the output.

process.out = cms.OutputModule("PoolOutputModule", 
    fileName = cms.untracked.string('PFMuMETevents.root') 
)


process.out.outputCommands = cms.untracked.vstring( 'drop *' )
process.out.outputCommands.extend(cms.untracked.vstring('keep *_genParticles_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_offlineBeamSpot_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_TriggerResults_*_HLT'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_hltTriggerSummaryAOD_*_HLT'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_muons_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_particleFlow_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_distortedMuons_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_distortedPFCand_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep recoTracks_globalMuons_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep recoTracks_standAloneMuons_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_met_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_corMetGlobalMuons_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_tcMet_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_pfMet_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_distpfMet_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_ak5CaloJets_*_*'))


process.outpath = cms.EndPath(process.out)




# And the logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True),
    Rethrow = cms.untracked.vstring('Unknown', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ModuleFailure', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FatalRootError', 
        'NotFound')
)

process.MessageLogger.cerr.FwkReport.reportEvery = 1


