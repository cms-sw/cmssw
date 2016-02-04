import FWCore.ParameterSet.Config as cms

process = cms.Process("PAT")



process.MessageLogger = cms.Service(
        "MessageLogger",
            categories = cms.untracked.vstring('info', 'debug','cout')
            )




# source
process.source = cms.Source("PoolSource", 
       fileNames = cms.untracked.vstring(
#'rfio:/castor/cern.ch/cms/store/caf/user/meridian/MinimumBias/BeamCommissioning09_EGMSkim/bb33bb16085462eaeb12c180f3bcafc3/EGMFirstCollisionSkim_123592_4.root',
    'file:/tmp/rompotis/bscFilter_123615_6.root',
#    'file:/tmp/rompotis/minbias_Summer09_STARTUP3X_V8D_900GeV_v1_1.root'
    ),                          
                            
     skipBadFiles = cms.untracked.bool(True),
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(200) )

# inputTagEnding = "EXPRESS"
# inputTagEnding = "EGMSKIM"
inputTagEnding = "RECO"



## Load additional RECO config
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('GR09_P_V7::All')
#process.GlobalTag.globaltag = cms.string('MC_31X_V5::All') #ideal conditions - 1e31 menu
#process.GlobalTag.globaltag = cms.string('STARTUP31X_V4::All') #ideal conditions - 8e29 menu

process.load("Configuration.StandardSequences.MagneticField_cff")

## Load necessary stuff for tcMET
# tracking geometry
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")

# load the necessary pat sequences
process.load("CommonTools.ParticleFlow.PF2PAT_cff")
process.load("PhysicsTools.PatAlgos.patSequences_cff")



process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)

## this is for the correct calculation of type1 MET
#from JetMETCorrections.Configuration.L2L3Corrections_Summer08Redigi_cff import *
#from JetMETCorrections.Type1MET.MetType1Corrections_cff import *
#process.load("JetMETCorrections.Type1MET.MetType1Corrections_cff")
#process.load("JetMETCorrections.Configuration.L2L3Corrections_Summer08Redigi_cff")

#process.metMuonJESCorSC5 = process.metJESCorSC5CaloJet.clone()
#process.metMuonJESCorSC5.inputUncorJetsLabel = "sisCone5CaloJets"
#process.metMuonJESCorSC5.corrector = "L2L3JetCorrectorSC5Calo"
#process.metMuonJESCorSC5.inputUncorMetLabel = "corMetGlobalMuons"
#process.metCorSequence = cms.Sequence(process.metMuonJESCorSC5)

# make this collection of type 1 corrected MET a pat collection for the code to handle
process.layer1mcMETs = process.layer1METs.clone(
    metSource = cms.InputTag("corMetGlobalMuons"),
    addTrigMatch = cms.bool(False),
    addMuonCorrections = cms.bool(False),
    addGenMET = cms.bool(False),
    )
process.layer1METs.addTrigMatch = cms.bool(False)
process.layer1METs.addGenMET = cms.bool(False)

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
process.allLayer1Electrons.isoDeposits = cms.PSet()
process.allLayer1Electrons.userIsolation = cms.PSet()
process.allLayer1Electrons.addElectronID = cms.bool(False)
process.allLayer1Electrons.electronIDSources = cms.PSet()
process.allLayer1Electrons.addGenMatch = cms.bool(False)
process.allLayer1Electrons.embedGenMatch = cms.bool(False)
process.allLayer1Electrons.embedHighLevelSelection = cms.bool(False)
##
process.allLayer1Muons.addGenMatch = cms.bool(False)
process.allLayer1Muons.embedGenMatch = cms.bool(False)
##
#process.makeAllLayer1Electrons = cms.Sequence(process.patElectronIsolation*process.allLayer1Electrons)
process.makeAllLayer1Electrons = cms.Sequence(process.allLayer1Electrons)
process.makeAllLayer1Muons = cms.Sequence(process.allLayer1Muons)
##
process.allLayer1Objects = cms.Sequence(process.makeAllLayer1Electrons+process.makeAllLayer1Muons+process.makeLayer1METs
                                        +process.layer1mcMETs)
process.selectedLayer1Objects = cms.Sequence(process.selectedLayer1Electrons+process.selectedLayer1Muons)
process.cleanLayer1Objects  = cms.Sequence(process.cleanLayer1Muons*process.cleanLayer1Electrons)
process.countLayer1Objects  = cms.Sequence(process.countLayer1Electrons+process.countLayer1Muons)

process.patDefaultSequence = cms.Sequence(process.allLayer1Objects * process.selectedLayer1Objects *
                                          process.cleanLayer1Objects*process.countLayer1Objects
                                          )

process.eca = cms.EDAnalyzer("EventContentAnalyzer")



process.rootskimmer = cms.EDAnalyzer(
    'GenPurposeSkimmerData',
# output file                   #######################################
    outputfile = cms.untracked.string('./bkg.root'),
    InputTagEnding = cms.untracked.string(inputTagEnding),
# collections
    ElectronCollection = cms.untracked.InputTag("selectedLayer1Electrons"),
    MetCollectionTag   = cms.untracked.InputTag( "met"),
    tcMetCollectionTag = cms.untracked.InputTag( "tcMet"),
    pfMetCollectionTag = cms.untracked.InputTag( "pfMet"),
#   genMetCollectionTag= cms.untracked.InputTag("genMetCalo", "", "HLT8E29"),
    t1MetCollectionTag = cms.untracked.InputTag("layer1METs"),
    mcMetCollectionTag = cms.untracked.InputTag("layer1mcMETs"),
    
# HLT ...............................................................    
    HLTCollectionE29 = cms.untracked.InputTag('hltTriggerSummaryAOD','','HLT'),
    HLTTriggerResultsE29 = cms.untracked.InputTag('TriggerResults','',inputTagEnding),
    # these are just for consistency with the old version
    HLTCollectionE31=cms.untracked.InputTag('hltTriggerSummaryAOD','','HLT'),    
    HLTTriggerResultsE31 = cms.untracked.InputTag('TriggerResults','','HLT'),
    ProbeHLTObjMaxDR = cms.untracked.double(0.1),

#   ECAL geometry   ###################################################
    BarrelMaxEta = cms.untracked.double(1.4442),
    EndcapMinEta = cms.untracked.double(1.56),
    EndcapMaxEta = cms.untracked.double(2.5),
# 
# some extra collections
    ctfTracksTag = cms.untracked.InputTag("generalTracks", "", "RECO"),
    corHybridsc = cms.untracked.InputTag("correctedHybridSuperClusters","", "RECO"),
    multi5x5sc = cms.untracked.InputTag("multi5x5SuperClustersWithPreshower","", "RECO"),
    )

#process.patDefaultSequence.remove(process.allLayer1Taus)



process.p = cms.Path(process.PF2PAT + process.patDefaultSequence + process.rootskimmer )
#process.p = cms.Path(process.PF2PAT + process.patDefaultSequence + process.eca )



