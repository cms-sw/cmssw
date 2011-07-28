import FWCore.ParameterSet.Config as cms

process = cms.Process("ZEEPLOTS")


process.MessageLogger = cms.Service(
        "MessageLogger",
            categories = cms.untracked.vstring('info', 'debug','cout')
            )

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)


# source
process.source = cms.Source("PoolSource", 
     #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/r/rompotis/RedigiSummer08RootTrees/WenuRedigi_RECO_SAMPLE.root')
     fileNames = cms.untracked.vstring(
    'file:zeeCandidates.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

## Load additional processes
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
## global tags:
process.GlobalTag.globaltag = cms.string('MC_31X_V5::All')
#process.GlobalTag.globaltag = cms.string('STARTUP31X_V4::All')
process.load("Configuration.StandardSequences.MagneticField_cff")



####################################################################################
##
## the Z selection that you prefer
selection_a2 = cms.PSet (
    trackIso_EB = cms.untracked.double(7.2),
    ecalIso_EB = cms.untracked.double(5.7),
    hcalIso_EB = cms.untracked.double(8.1),
    sihih_EB = cms.untracked.double(0.01),
    dphi_EB = cms.untracked.double(1000.),
    deta_EB = cms.untracked.double(0.0071),
    hoe_EB = cms.untracked.double(1000),
    
    trackIso_EE = cms.untracked.double(5.1),
    ecalIso_EE = cms.untracked.double(5.0),
    hcalIso_EE = cms.untracked.double(3.4),
    sihih_EE = cms.untracked.double(0.028),
    dphi_EE = cms.untracked.double(1000.),
    deta_EE = cms.untracked.double(0.0066),
    hoe_EE = cms.untracked.double(1000.)
    )

selection_inverse = cms.PSet (
    trackIso_EB_inv = cms.untracked.bool(True),
    trackIso_EE_inv = cms.untracked.bool(True)
    )

####################################################################################
##
## and the plot creator
process.plotter = cms.EDAnalyzer('ZeePlots',
                                 selection_a2,
                                 zeeCollectionTag = cms.untracked.InputTag("zeeFilter","selectedZeeCandidates","PAT")
                                 )




process.p = cms.Path(process.plotter)


