import FWCore.ParameterSet.Config as cms

process = cms.Process("PAT")



process.MessageLogger = cms.Service(
        "MessageLogger",
            categories = cms.untracked.vstring('info', 'debug','cout')
            )




# source
process.source = cms.Source("PoolSource", 
     # fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/r/rompotis/RedigiSummer08RootTrees/WenuRedigi_RECO_SAMPLE.root')
#     fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/r/rompotis/Summer09Studies/zee_Summer09-MC_31X_V3_AODSIM_v1_AODSIM.root')
     fileNames = cms.untracked.vstring('file:zee_Summer09-MC_31X_V3_AODSIM_v1_AODSIM.root'),
     #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/r/rompotis/Summer09Studies/QCD_EMEnriched_Pt30to80_AODSIM_7E27C8EA-7984-DE11-BA59-00151796C158.root')
     skipBadFiles = cms.untracked.bool(True),
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )


## Load additional RECO config
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = cms.string('MC_31X_V5::All') #ideal conditions - 1e31 menu
process.GlobalTag.globaltag = cms.string('STARTUP31X_V4::All') #ideal conditions - 8e29 menu

process.load("Configuration.StandardSequences.MagneticField_cff")

## Load necessary stuff for tcMET
# tracking geometry
process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")

# load the necessary pat sequences
process.load("PhysicsTools.PFCandProducer.PF2PAT_cff")
process.load("PhysicsTools.PatAlgos.patSequences_cff")



process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)

## this is for the correct calculation of type1 MET
#from JetMETCorrections.Configuration.L2L3Corrections_Summer08Redigi_cff import *
#from JetMETCorrections.Type1MET.MetType1Corrections_cff import *
process.load("JetMETCorrections.Type1MET.MetType1Corrections_cff")
process.load("JetMETCorrections.Configuration.L2L3Corrections_Summer08Redigi_cff")

process.metMuonJESCorSC5 = process.metJESCorSC5CaloJet.clone()
process.metMuonJESCorSC5.inputUncorJetsLabel = "sisCone5CaloJets"
process.metMuonJESCorSC5.corrector = "L2L3JetCorrectorSC5Calo"
process.metMuonJESCorSC5.inputUncorMetLabel = "caloMetM"

process.metCorSequence = cms.Sequence(process.metMuonJESCorSC5)

# make this collection of type 1 corrected MET a pat collection for the code to handle
process.layer1TwikiT1METs = process.layer1METs.clone(
    metSource = cms.InputTag("metMuonJESCorSC5","","PAT"),
    addTrigMatch = cms.bool(False),
    addMuonCorrections = cms.bool(False),
    addGenMET = cms.bool(False),
    )


#
# the explicit declaration of rechit collections is just for compatibility with the header
# version of 312 - proper Tags have no need of that
#
# for ecal isolation: replace the ECAL rechit collection
process.eleIsoDepositEcalFromHits.ExtractorPSet.barrelEcalHits = cms.InputTag("reducedEcalRecHitsEB", "", "RECO")
process.eleIsoDepositEcalFromHits.ExtractorPSet.endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE", "", "RECO")
#
#
process.eidRobustHighEnergy.reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB", "", "RECO")
process.eidRobustHighEnergy.reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE", "", "RECO")
#
### create the good old ecal isolation for EE ##########

##
##  this is how to compute isolation yourself for testing purposes
##
#process.load("RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff")
#process.electronEcalRecHitIsolationLcone.ecalBarrelRecHitProducer = cms.InputTag("reducedEcalRecHitsEB")
#process.electronEcalRecHitIsolationLcone.ecalEndcapRecHitProducer = cms.InputTag("reducedEcalRecHitsEB")
#process.electronEcalRecHitIsolationLcone.ecalBarrelRecHitCollection = cms.InputTag("")
#process.electronEcalRecHitIsolationLcone.ecalEndcapRecHitCollection = cms.InputTag("")
#process.electronEcalRecHitIsolationScone.ecalBarrelRecHitProducer = cms.InputTag("reducedEcalRecHitsEB")
#process.electronEcalRecHitIsolationScone.ecalEndcapRecHitProducer = cms.InputTag("reducedEcalRecHitsEB")
#process.electronEcalRecHitIsolationScone.ecalBarrelRecHitCollection = cms.InputTag("")
#process.electronEcalRecHitIsolationScone.ecalEndcapRecHitCollection = cms.InputTag("")
#process.patElectronIsolation = process.egammaIsolationSequence


########################################################
#from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import allLayer1Electrons
# add the user iso
#
# NOTE!!! Egamma Recommendations Track Iso: Scone (0.3), ecal+hcal Lcone (0.4)
##               for further studies the rest will be set as user isolations
#
process.allLayer1Electrons.isolation = cms.PSet()

##
process.allLayer1Objects = cms.Sequence(process.makeAllLayer1Electrons+process.makeAllLayer1Muons+process.makeLayer1METs+process.layer1TwikiT1METs)
process.selectedLayer1Objects = cms.Sequence(process.selectedLayer1Electrons+process.selectedLayer1Muons)
process.cleanLayer1Objects  = cms.Sequence(process.cleanLayer1Muons*process.cleanLayer1Electrons)
process.countLayer1Objects  = cms.Sequence(process.countLayer1Electrons+process.countLayer1Muons)

process.patDefaultSequence = cms.Sequence(process.allLayer1Objects * process.selectedLayer1Objects *
                                          process.cleanLayer1Objects*process.countLayer1Objects
                                          )

process.eca = cms.EDAnalyzer("EventContentAnalyzer")



process.rootskimmer = cms.EDAnalyzer(
    'GenPurposeSkimmer',
# output file                   #######################################
    outputfile = cms.untracked.string('./bkg.root'),
# collections
    ElectronCollection = cms.untracked.InputTag("selectedLayer1Electrons"),
    MetCollectionTag   = cms.untracked.InputTag( "met","","RECO"),
    tcMetCollectionTag = cms.untracked.InputTag( "tcMet"),
    pfMetCollectionTag = cms.untracked.InputTag( "pfMet"),
    genMetCollectionTag= cms.untracked.InputTag("genMetCalo", "", "HLT8E29"),
    t1MetCollectionTag = cms.untracked.InputTag("layer1METs"),
    t1MetCollectionTagTwiki = cms.untracked.InputTag("layer1TwikiT1METs"),
    
# HLT ...............................................................    
    HLTCollectionE29 = cms.untracked.InputTag('hltTriggerSummaryAOD','','HLT8E29'),
    HLTCollectionE31=cms.untracked.InputTag('hltTriggerSummaryAOD','','HLT'),
    HLTTriggerResultsE29 = cms.untracked.InputTag('TriggerResults','','HLT8E29'),
    HLTTriggerResultsE31 = cms.untracked.InputTag('TriggerResults','','HLT'),
    ProbeHLTObjMaxDR = cms.untracked.double(0.1),

#   ECAL geometry   ###################################################
    BarrelMaxEta = cms.untracked.double(1.4442),
    EndcapMinEta = cms.untracked.double(1.56),
    EndcapMaxEta = cms.untracked.double(2.5),

# some MC information
    MCCollection = cms.untracked.InputTag("genParticles", "", "HLT8E29"),
    # deta and dphi have default values and there is no reason to change them

    )

#process.patDefaultSequence.remove(process.allLayer1Taus)



process.p = cms.Path(process.metCorSequence + process.PF2PAT + process.patDefaultSequence + process.rootskimmer )
#process.p = cms.Path(process.PF2PAT + process.patDefaultSequence + process.eca )



