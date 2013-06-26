import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("distortedMuMET")

process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring(
#        'file:/ciet3b/data4/Spring10_10invpb_AODRED/Wmunu/Wmunu_1.root'
      '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEFC70B6-F53D-DF11-B57E-003048679150.root'
      )
)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('distortedMuons'),
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
            #threshold = cms.untracked.string('INFO')
            threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)


# GEN-REC muon matching
process.genMatchMap = cms.EDFilter("MCTruthDeltaRMatcherNew",
    src = cms.InputTag("muons"),
    matched = cms.InputTag("genParticles"),
#    matched = cms.InputTag("prunedGenParticles"),
    distMin = cms.double(0.15),
    matchPDGId = cms.vint32(13)
)

# Create a new "distorted" Muon collection
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
# mm  now add distorted MET stuff
# mm


# mm
# mm first distort Calo
# mm

process.load("FWCore.MessageService.MessageLogger_cfi")
## configure geometry
process.load("Configuration.StandardSequences.Geometry_cff")
## configure B field
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")
process.GlobalTag.globaltag = 'START3X_V26::All'

process.load("ElectroWeakAnalysis.Utilities.distMuonMETValueMapProducer_cff")
process.load("ElectroWeakAnalysis.Utilities.MetdistMuonCorrections_cff")

# mm
# mm then distort TC
# mm

process.load("ElectroWeakAnalysis.Utilities.distMuonTCMETValueMapProducer_cff")
process.load("ElectroWeakAnalysis.Utilities.distTCMET_cfi")


process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('distMuonMETevents.root')
)
process.out.outputCommands = cms.untracked.vstring( 'drop *' )
process.out.outputCommands.extend(cms.untracked.vstring('keep *_genParticles_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_offlineBeamSpot_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_TriggerResults_*_HLT'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_hltTriggerSummaryAOD_*_HLT'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_muons_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_distortedMuons_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep recoTracks_globalMuons_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep recoTracks_standAloneMuons_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_met_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_corMetGlobalMuons_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_distMetGlobalMuons_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_tcMet_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_disttcMet_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_pfMet_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_ak5CaloJets_*_*'))



# Steering the process
process.distortedMuMET = cms.Path(
        process.genMatchMap+process.distortedMuons
      +process.distmuonMETValueMapProducer+process.distMetGlobalMuons
      +process.distmuonTCMETValueMapProducer+process.disttcMet
#      *process.selectCaloMetWMuNus
)

process.end = cms.EndPath(process.out)
#process.end = cms.EndPath(process.wmnOutput)

