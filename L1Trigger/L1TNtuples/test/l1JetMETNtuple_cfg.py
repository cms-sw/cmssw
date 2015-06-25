import FWCore.ParameterSet.Config as cms

from L1TriggerDPG.L1Ntuples.l1Ntuple_cfg import *

# global tag

process.GlobalTag.globaltag = cms.string('GR_R_52_V7::All')

# load standard RECO for MET cleaning
process.load('Configuration.StandardSequences.Reconstruction_cff')

##################################good collisions############################################
# This filter select ~73% of events in MinBias PD
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

process.L1T1coll=process.hltLevel1GTSeed.clone()
process.L1T1coll.L1TechTriggerSeeding = cms.bool(True)
process.L1T1coll.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')

process.primaryVertexFilter = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"), # tracksSize() > 3 for the older cut
   filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)


process.noscraping = cms.EDFilter("FilterOutScraping",
  applyfilter = cms.untracked.bool(True),
  debugOn = cms.untracked.bool(False),
  numtrack = cms.untracked.uint32(10),
  thresh = cms.untracked.double(0.25)
)

### HLT Filter ###

# HCAL noise filter
process.load('CommonTools/RecoAlgos/HBHENoiseFilter_cfi')
process.hbhefilter = cms.Path(process.HBHENoiseFilter)


# L1 ntuples
process.l1NtupleProducer.generatorSource      = cms.InputTag("none")
process.l1NtupleProducer.simulationSource     = cms.InputTag("none")
process.l1NtupleProducer.gmtSource            = cms.InputTag("gtDigis")
process.l1NtupleProducer.gtEvmSource          = cms.InputTag("none")
process.l1NtupleProducer.gtSource             = cms.InputTag("gtDigis")
process.l1NtupleProducer.gctIsoEmSource       = cms.InputTag("gctDigis","isoEm")
process.l1NtupleProducer.gctNonIsoEmSource    = cms.InputTag("gctDigis","nonIsoEm")
process.l1NtupleProducer.gctCentralJetsSource = cms.InputTag("gctDigis","cenJets")
process.l1NtupleProducer.gctTauJetsSource     = cms.InputTag("gctDigis","tauJets")
process.l1NtupleProducer.gctForwardJetsSource = cms.InputTag("gctDigis","forJets")
process.l1NtupleProducer.gctEnergySumsSource  = cms.InputTag("gctDigis")
process.l1NtupleProducer.rctSource            = cms.InputTag("gctDigis")
process.l1NtupleProducer.dttfSource           = cms.InputTag("none")
process.l1NtupleProducer.csctfTrkSource       = cms.InputTag("none")
process.l1NtupleProducer.csctfLCTSource       = cms.InputTag("none")
process.l1NtupleProducer.csctfStatusSource    = cms.InputTag("none")

process.l1RecoTreeProducer.jetptThreshold = cms.double(5)

process.gctDigis.numberOfGctSamplesToUnpack = cms.uint32(5)
process.l1extraParticles.centralBxOnly = cms.bool(False)

process.p.remove(process.dttfDigis)
process.p.remove(process.csctfDigis)
process.p.remove(process.l1extraParticles)
process.p.remove(process.l1ExtraTreeProducer)
process.p.remove(process.l1MuonRecoTreeProducer)
#from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector

#process.goodOfflinePrimaryVertices = cms.EDFilter(
#        "PrimaryVertexObjectFilter",
#        filterParams = pvSelector.clone( minNdof = cms.double(4.0), maxZ = cms.double(24.0) ),
#        src=cms.InputTag('offlinePrimaryVertices')
#        )
#from PhysicsTools.PatAlgos.tools.helpers import listModules, applyPostfix
#process.load("PhysicsTools.PatAlgos.patSequences_cff")

#from PhysicsTools.PatAlgos.tools.pfTools import *
#postfix = "PFlow"
#usePF2PAT(process,runPF2PAT=True, jetAlgo='AK5', runOnMC=False, postfix=postfix)
#process.pfPileUpPFlow.Enable = True
#process.pfPileUpPFlow.Vertices = 'goodOfflinePrimaryVertices'
#process.pfJetsPFlow.doAreaFastjet = True
#process.pfJetsPFlow.doRhoFastjet = True
#process.patJetCorrFactorsPFlow.rho = cms.InputTag("kt6PFJetsPFlow", "rho")


#process.pfPileUpPFlow.checkClosestZVertex = cms.bool(False)
#from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets
#process.kt6PFJetsPFlow = kt4PFJets.clone(
#            rParam = cms.double(0.6),
#            src = cms.InputTag(postfix),
#            doAreaFastjet = cms.bool(True),
#            doRhoFastjet = cms.bool(True)
#            )
#getattr(process,"patPF2PATSequence"+postfix).replace( getattr(process,"pfNoElectron"+postfix), getattr(process,"pfNoElectron"+postfix)*process.kt6PFJetsPFlow )
#process.patseq = cms.Sequence(    
#        process.goodOfflinePrimaryVertices*
#        getattr(process,"patPF2PATSequence"+postfix)
#        )
#override the default global tag!
#process.load("L1TriggerConfig.GctConfigProducers.l1GctConfig_cfi")
#Dump the config info
#process.load("L1TriggerConfig.GctConfigProducers.l1GctConfigDump_cfi")




process.ntuple = cms.Path(
#    process.L1T1coll
   # process.HLTFilter
#    +process.primaryVertexFilter
#    +process.noscraping
    process.l1NtupleProducer
    +process.HBHENoiseFilter
    +process.l1ExtraTreeProducer
    +process.l1RecoTreeProducer
)
process.MessageLogger = cms.Service(
    "MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout         = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG'))
  )



# jet corrections
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
process.ak5CaloL1Offset.useCondDB = False
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )


readFiles.extend( [
        '/store/data/Run2011B/HT/AOD/19Nov2011-v1/0000/3ACD091B-281E-E111-894B-00261894386D.root'
] )
