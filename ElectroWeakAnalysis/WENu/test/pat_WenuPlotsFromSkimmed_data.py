import FWCore.ParameterSet.Config as cms

process = cms.Process("WENUPLOTS")


process.MessageLogger = cms.Service(
        "MessageLogger",
            categories = cms.untracked.vstring('info', 'debug','cout')
            )

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)


# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:wenuCandidates.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

## Load additional processes
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
## global tags:
process.GlobalTag.globaltag = cms.string('GR09_P_V7::All')
#process.GlobalTag.globaltag = cms.string('MC_31X_V5::All')
#process.GlobalTag.globaltag = cms.string('STARTUP31X_V4::All')
process.load("Configuration.StandardSequences.MagneticField_cff")


################################################################################################
###    P r e p a r a t i o n      o f    t h e    P A T    O b j e c t s   f r o m    A O D  ###
################################################################################################

####################################################################################
##
## the W selection that you prefer
selection_a2 = cms.PSet (
    trackIso_EB = cms.untracked.double(2.2),
    ecalIso_EB = cms.untracked.double(4.2),
    hcalIso_EB = cms.untracked.double(2.0),
    sihih_EB = cms.untracked.double(0.0099),
    dphi_EB = cms.untracked.double(0.025),
    deta_EB = cms.untracked.double(0.0040),
    hoe_EB = cms.untracked.double(1000.0),
    
    trackIso_EE = cms.untracked.double(1.1),
    ecalIso_EE = cms.untracked.double(3.4),
    hcalIso_EE = cms.untracked.double(1.3),
    sihih_EE = cms.untracked.double(0.028),
    dphi_EE = cms.untracked.double(0.020),
    deta_EE = cms.untracked.double(0.0066),
    hoe_EE = cms.untracked.double(1000.0)
    )

selection_test = cms.PSet (
    trackIso_EB = cms.untracked.double(10),
    ecalIso_EB = cms.untracked.double(10),
    hcalIso_EB = cms.untracked.double(10),
    sihih_EB = cms.untracked.double(0.1),
    dphi_EB = cms.untracked.double(1),
    deta_EB = cms.untracked.double(1),
    hoe_EB = cms.untracked.double(1),
    
    trackIso_EE = cms.untracked.double(10),
    ecalIso_EE = cms.untracked.double(10),
    hcalIso_EE = cms.untracked.double(10),
    sihih_EE = cms.untracked.double(1),
    dphi_EE = cms.untracked.double(1),
    deta_EE = cms.untracked.double(1),
    hoe_EE = cms.untracked.double(1)
    )

selection_inverse = cms.PSet (
    trackIso_EB_inv = cms.untracked.bool(True),
    trackIso_EE_inv = cms.untracked.bool(True)
    )

####################################################################################
##
## and the plot creator
process.plotter = cms.EDAnalyzer('WenuPlots',
                                 selection_test,
                                 selection_inverse,
                                 wenuCollectionTag = cms.untracked.InputTag(
    "wenuFilter","selectedWenuCandidates","PAT"),
                                 outputFile = cms.untracked.string("myHistos.root"),
                                 )




process.p = cms.Path(process.plotter)
# process.p = cms.Path(process.patSequences + process.wenuFilter + process.eca)

