#
#  SUSY-PAT configuration file
#
#  PAT configuration for the SUSY group - 3X series
#  More information here:
#  https://twiki.cern.ch/twiki/bin/view/CMS/SusyPatLayer1DefV6
#

# Starting with a skeleton process which gets imported with the following line
from PhysicsTools.PatAlgos.patTemplate_cfg import *

#-- Meta data to be logged in DBS ---------------------------------------------
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.13 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/PhysicsTools/Configuration/test/SUSY_pattuple_cfg.py,v $'),
    annotation = cms.untracked.string('SUSY pattuple definition')
)

#-- Message Logger ------------------------------------------------------------
process.MessageLogger.categories.append('PATSummaryTables')
process.MessageLogger.cerr.PATSummaryTables = cms.untracked.PSet(
    limit = cms.untracked.int32(-1),
    reportEvery = cms.untracked.int32(1)
    )
process.MessageLogger.cerr.FwkReport.reportEvery = 100

#-- Input Source --------------------------------------------------------------
process.source.fileNames = [
    'rfio://?svcclass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/fronga/V6production/PYTHIA6_SUSY_LM0_sftsht_10TeV_cff_py_RAW2DIGI_RECO_1.root'
    ]
process.maxEvents.input = -1
# Due to problem in production of LM samples: same event number appears multiple times
process.source.duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

#-- Calibration tag -----------------------------------------------------------
# Should match input file's tag
process.GlobalTag.globaltag = 'STARTUP31X_V1::All'

#-- PAT standard config -------------------------------------------------------
process.load("PhysicsTools.PatAlgos.patSequences_cff")

#-- Changes for electron and photon ID ----------------------------------------
# Turn off photon-electron cleaning (i.e., flag only)
process.cleanLayer1Photons.checkOverlaps.electrons.requireNoOverlaps = False
# Remove embedding of superClusters, will keep entire superCluster collection
process.allLayer1Electrons.embedSuperCluster = False
process.allLayer1Photons.embedSuperCluster   = False

#-- Tuning of Monte Carlo matching --------------------------------------------
# Also match with leptons of opposite charge
process.electronMatch.checkCharge = False
process.muonMatch.checkCharge     = False
process.tauMatch.checkCharge      = False

#-- Jet plus tracks -----------------------------------------------------------
process.load("JetMETCorrections.Configuration.JetCorrectionsRecord_cfi")

# ---------- ESSources delivering ZSP correctors
process.ZSPJetCorrectorIcone5 = cms.ESSource( "ZSPJetCorrectionService", tagName = cms.string('ZSP_CMSSW219_Iterative_Cone_05'), label = cms.string('ZSPJetCorrectorIcone5'))
process.ZSPJetCorrectorSiscone5 = cms.ESSource( "ZSPJetCorrectionService", tagName = cms.string('ZSP_CMSSW219_Iterative_Cone_05'), label = cms.string('ZSPJetCorrectorSiscone5'))
process.ZSPJetCorrectorAntiKt5 = cms.ESSource( "ZSPJetCorrectionService", tagName = cms.string('ZSP_CMSSW219_Iterative_Cone_05'), label = cms.string('ZSPJetCorrectorAntiKt5'))
# ---------- EDProducers using ZSP correctors
process.ZSPJetCorJetIcone5 = cms.EDProducer( "CaloJetCorrectionProducer", src = cms.InputTag("iterativeCone5CaloJets"), correctors = cms.vstring('ZSPJetCorrectorIcone5'), alias = cms.untracked.string('ZSPJetCorJetIcone5'))
process.ZSPJetCorJetSiscone5 = cms.EDProducer( "CaloJetCorrectionProducer", src = cms.InputTag("sisCone5CaloJets"), correctors = cms.vstring('ZSPJetCorrectorSiscone5'), alias = cms.untracked.string('ZSPJetCorJetSiscone5'))
process.ZSPJetCorJetAntiKt5 = cms.EDProducer( "CaloJetCorrectionProducer", src = cms.InputTag("antikt5CaloJets"), correctors = cms.vstring('ZSPJetCorrectorAntiKt5'), alias = cms.untracked.string('ZSPJetCorJetAntiKt5'))
# ---------- Jet-track association for IC5, SC5 and AK5
process.load("RecoJets.Configuration.RecoJetAssociations_cff")
process.load("RecoJets.JetAssociationProducers.iterativeCone5JTA_cff")

process.ZSPiterativeCone5JetTracksAssociatorAtVertex = process.iterativeCone5JetTracksAssociatorAtVertex.clone() 
process.ZSPiterativeCone5JetTracksAssociatorAtVertex.jets = cms.InputTag("ZSPJetCorJetIcone5")
process.ZSPiterativeCone5JetTracksAssociatorAtCaloFace = process.iterativeCone5JetTracksAssociatorAtCaloFace.clone()
process.ZSPiterativeCone5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("ZSPJetCorJetIcone5")
process.ZSPiterativeCone5JetExtender = process.iterativeCone5JetExtender.clone() 
process.ZSPiterativeCone5JetExtender.jets = cms.InputTag("ZSPJetCorJetIcone5")
process.ZSPiterativeCone5JetExtender.jet2TracksAtCALO = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtCaloFace")
process.ZSPiterativeCone5JetExtender.jet2TracksAtVX = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtVertex")

process.ZSPSisCone5JetTracksAssociatorAtVertex = process.iterativeCone5JetTracksAssociatorAtVertex.clone()
process.ZSPSisCone5JetTracksAssociatorAtVertex.jets = cms.InputTag("ZSPJetCorJetSiscone5")
process.ZSPSisCone5JetTracksAssociatorAtCaloFace = process.iterativeCone5JetTracksAssociatorAtCaloFace.clone()
process.ZSPSisCone5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("ZSPJetCorJetSiscone5")
process.ZSPSisCone5JetExtender = process.iterativeCone5JetExtender.clone()
process.ZSPSisCone5JetExtender.jets = cms.InputTag("ZSPJetCorJetSiscone5")
process.ZSPSisCone5JetExtender.jet2TracksAtCALO = cms.InputTag("ZSPSisCone5JetTracksAssociatorAtCaloFace")
process.ZSPSisCone5JetExtender.jet2TracksAtVX = cms.InputTag("ZSPSisCone5JetTracksAssociatorAtVertex")

process.ZSPAntiKt5JetTracksAssociatorAtVertex = process.iterativeCone5JetTracksAssociatorAtVertex.clone()
process.ZSPAntiKt5JetTracksAssociatorAtVertex.jets = cms.InputTag("ZSPJetCorJetAntiKt5")
process.ZSPAntiKt5JetTracksAssociatorAtCaloFace = process.iterativeCone5JetTracksAssociatorAtCaloFace.clone()
process.ZSPAntiKt5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("ZSPJetCorJetAntiKt5")
process.ZSPAntiKt5JetExtender = process.iterativeCone5JetExtender.clone()
process.ZSPAntiKt5JetExtender.jets = cms.InputTag("ZSPJetCorJetAntiKt5")
process.ZSPAntiKt5JetExtender.jet2TracksAtCALO = cms.InputTag("ZSPAntiKt5JetTracksAssociatorAtCaloFace")
process.ZSPAntiKt5JetExtender.jet2TracksAtVX = cms.InputTag("ZSPAntiKt5JetTracksAssociatorAtVertex")

# ---------- ESSources delivering JPT correctors
from JetMETCorrections.Configuration.JetPlusTrackCorrections_cfi import *
process.JetPlusTrackZSPCorrectorIcone5 = cms.ESSource( "JetPlusTrackCorrectionService", cms.PSet(JPTZSPCorrectorICone5), label = cms.string('JetPlusTrackZSPCorrectorIcone5'),)
process.JetPlusTrackZSPCorrectorIcone5.JetTrackCollectionAtVertex = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtVertex")
process.JetPlusTrackZSPCorrectorIcone5.JetTrackCollectionAtCalo = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtCaloFace")
process.JetPlusTrackZSPCorrectorIcone5.SplitMergeP = cms.int32(0)
process.JetPlusTrackZSPCorrectorIcone5.eIDValueMap = cms.InputTag("eidTight")

process.JetPlusTrackZSPCorrectorSiscone5 = cms.ESSource( "JetPlusTrackCorrectionService", cms.PSet(JPTZSPCorrectorICone5), label = cms.string('JetPlusTrackZSPCorrectorSiscone5'),)
process.JetPlusTrackZSPCorrectorSiscone5.JetTrackCollectionAtVertex = cms.InputTag("ZSPSisCone5JetTracksAssociatorAtVertex")
process.JetPlusTrackZSPCorrectorSiscone5.JetTrackCollectionAtCalo = cms.InputTag("ZSPSisCone5JetTracksAssociatorAtCaloFace")
process.JetPlusTrackZSPCorrectorSiscone5.SplitMergeP = cms.int32(1)
process.JetPlusTrackZSPCorrectorSiscone5.eIDValueMap = cms.InputTag("eidTight")

process.JetPlusTrackZSPCorrectorAntiKt5 = cms.ESSource( "JetPlusTrackCorrectionService", cms.PSet(JPTZSPCorrectorICone5), label = cms.string('JetPlusTrackZSPCorrectorAntiKt5'),)
process.JetPlusTrackZSPCorrectorAntiKt5.JetTrackCollectionAtVertex = cms.InputTag("ZSPAntiKt5JetTracksAssociatorAtVertex")
process.JetPlusTrackZSPCorrectorAntiKt5.JetTrackCollectionAtCalo = cms.InputTag("ZSPAntiKt5JetTracksAssociatorAtCaloFace")
process.JetPlusTrackZSPCorrectorAntiKt5.SplitMergeP = cms.int32(2)
process.JetPlusTrackZSPCorrectorAntiKt5.eIDValueMap = cms.InputTag("eidTight")

# ---------- EDProducers using JPT correctors
process.JetPlusTrackZSPCorJetIcone5 = cms.EDProducer( "CaloJetCorrectionProducer", src = cms.InputTag("ZSPJetCorJetIcone5"), correctors = cms.vstring('JetPlusTrackZSPCorrectorIcone5'), alias = cms.untracked.string('JetPlusTrackZSPCorJetIcone5'))
process.JetPlusTrackZSPCorJetSiscone5 = cms.EDProducer( "CaloJetCorrectionProducer", src = cms.InputTag("ZSPJetCorJetSiscone5"), correctors = cms.vstring('JetPlusTrackZSPCorrectorSiscone5'), alias = cms.untracked.string('JetPlusTrackZSPCorJetSiscone5'))
process.JetPlusTrackZSPCorJetAntiKt5 = cms.EDProducer( "CaloJetCorrectionProducer", src = cms.InputTag("ZSPJetCorJetAntiKt5"), correctors = cms.vstring('JetPlusTrackZSPCorrectorAntiKt5'), alias = cms.untracked.string('JetPlusTrackZSPCorJetAntiKt5'))

# ---------- Sequences
process.JetPlusTrackCorrectionsIcone5 = cms.Sequence(process.ZSPJetCorJetIcone5*process.ZSPiterativeCone5JetTracksAssociatorAtVertex*process.ZSPiterativeCone5JetTracksAssociatorAtCaloFace*process.ZSPiterativeCone5JetExtender*process.JetPlusTrackZSPCorJetIcone5)
process.JetPlusTrackCorrectionsSisCone5 = cms.Sequence(process.ZSPJetCorJetSiscone5*process.ZSPSisCone5JetTracksAssociatorAtVertex*process.ZSPSisCone5JetTracksAssociatorAtCaloFace*process.ZSPSisCone5JetExtender*process.JetPlusTrackZSPCorJetSiscone5)
process.JetPlusTrackCorrectionsAntiKt5 = cms.Sequence(process.ZSPJetCorJetAntiKt5*process.ZSPAntiKt5JetTracksAssociatorAtVertex*process.ZSPAntiKt5JetTracksAssociatorAtCaloFace*process.ZSPAntiKt5JetExtender*process.JetPlusTrackZSPCorJetAntiKt5)
process.JetPlusTrackCorrections = cms.Sequence(process.JetPlusTrackCorrectionsIcone5*process.JetPlusTrackCorrectionsSisCone5*process.JetPlusTrackCorrectionsAntiKt5)

#-- Extra Jet/MET collections -------------------------------------------------
from PhysicsTools.PatAlgos.tools.jetTools import *
# Add a few jet collections...
addJetCollection(process, cms.InputTag('antikt5CaloJets'),
                 'AK5',
                 doJTA            = True,
                 doBTagging       = True,
                 jetCorrLabel     = ('AK5','Calo'),
                 doType1MET       = True,
                 genJetCollection = cms.InputTag("antikt5GenJets")
                 )
addJetCollection(process,cms.InputTag('iterativeCone5PFJets'),
                 'IC5PF',
                 doJTA        = True,
                 doBTagging   = True,
                 jetCorrLabel = None,
                 doType1MET   = True,
                 doL1Cleaning = True,
                 doL1Counters = True,
                 genJetCollection=cms.InputTag("iterativeCone5GenJets")
                 )
addJetCollection(process,cms.InputTag('sisCone5CaloJets'),
                 'SC5',
                 doJTA        = True,
                 doBTagging   = True,
                 jetCorrLabel = ('SC5','Calo'),
                 doType1MET   = True,
                 doL1Cleaning = True,
                 doL1Counters = True,
                 genJetCollection=cms.InputTag("sisCone5GenJets")
                 )
# Load JPT sequence
addJetCollection(process,cms.InputTag('JetPlusTrackZSPCorJetIcone5'),
                 'IC5JPT',
                 doJTA        = True,
                 doBTagging   = True,
                 jetCorrLabel = None,
                 doType1MET   = False,
                 doL1Cleaning = True,
                 doL1Counters = True,
                 genJetCollection = cms.InputTag("iterativeCone5GenJets")
                 )
addJetCollection(process,cms.InputTag('JetPlusTrackZSPCorJetSiscone5'),
                 'SC5JPT',
                 doJTA        = True,
                 doBTagging   = True,
                 jetCorrLabel = None,
                 doType1MET   = False,
                 doL1Cleaning = True,
                 doL1Counters = True,
                 genJetCollection = cms.InputTag("sisCone5GenJets")
                 )
addJetCollection(process,cms.InputTag('JetPlusTrackZSPCorJetAntiKt5'),
                  'AK5JPT',
                  doJTA        = True,
                  doBTagging   = True,
                  jetCorrLabel = None,
                  doType1MET   = False,
                  doL1Cleaning = True,
                  doL1Counters = True,
                  genJetCollection = cms.InputTag("antikt5GenJets")
                  )

# Add tcMET and PFMET
from PhysicsTools.PatAlgos.tools.metTools import *
addTcMET(process,'TC')
addPfMET(process,'PF')

# Add latest HcalNoiseSummary
process.load("RecoMET.METProducers.hcalnoiseinfoproducer_cfi")
process.hcalnoise.refillRefVectors = True
process.hcalnoise.hcalNoiseRBXCollName = "hcalnoise" # This has changed in 33X

#-- Track Jets ----------------------------------------------------------------
# Select tracks for track jets
process.load("PhysicsTools.RecoAlgos.TrackWithVertexSelector_cfi")
process.trackWithVertexSelector.src              = cms.InputTag("generalTracks")
process.trackWithVertexSelector.ptMax            = cms.double(500.0) 
process.trackWithVertexSelector.normalizedChi2   = cms.double(100.0)
process.trackWithVertexSelector.vertexTag        = cms.InputTag("offlinePrimaryVertices")
process.trackWithVertexSelector.copyTrajectories = cms.untracked.bool(False)
process.trackWithVertexSelector.vtxFallback      = cms.bool(False)
process.trackWithVertexSelector.useVtx           = cms.bool(False)
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.tracksForJets = cms.EDProducer("ConcreteChargedCandidateProducer",
                                       src = cms.InputTag("trackWithVertexSelector"),
                                       particleType = cms.string('pi+')
                                       )
# Add jet collections
from RecoJets.JetProducers.SISConeJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
process.SISCone5TrackJets = cms.EDProducer("SISConeJetProducer",
                                           SISConeJetParameters,
                                           FastjetNoPU,
                                           src = cms.InputTag("tracksForJets"),
                                           jetType = cms.untracked.string('BasicJet'),
                                           alias = cms.untracked.string('SISCone5TrackJets'),
                                           coneRadius = cms.double(0.5),
                                           jetPtMin = cms.double(0.3),
                                           inputEMin = cms.double(0.0),
                                           inputEtMin = cms.double(0.2),
                                           )
process.addTrackJets = cms.Sequence(  process.trackWithVertexSelector
                                    * process.tracksForJets
                                    * process.SISCone5TrackJets )
addJetCollection(process,cms.InputTag('SISCone5TrackJets'),
                 'SC5Track',
                 doJTA        = False,
                 doBTagging   = True,
                 jetCorrLabel = None,
                 doType1MET   = False,
                 doL1Cleaning = True,
                 doL1Counters = True,
                 genJetCollection = cms.InputTag("sisCone5GenJets")
                 )

#-- Tune contents of jet collections  -----------------------------------------
for jetName in ( '', 'AK5', 'IC5PF', 'SC5', 'SC5Track', 'AK5JPT', 'SC5JPT' ):
    module = getattr(process,'allLayer1Jets'+jetName)
    module.addTagInfos = False    # Remove tag infos
    module.addJetID    = True     # Add JetID variables
    module.embedGenJetMatch = False # Only keep reference, since we anyway keep the genJet collections

#-- Output module configuration -----------------------------------------------
from PhysicsTools.PatAlgos.patEventContent_cff import patEventContent
process.out.fileName = 'file://./test.root'       # <-- CHANGE THIS TO SUIT YOUR NEEDS

# Custom settings
process.out.splitLevel = cms.untracked.int32(99)  # Turn on split level (smaller files???)
process.out.overrideInputFileSplitLevels = cms.untracked.bool(True)
process.out.dropMetaData = cms.untracked.string('DROPPED')   # Get rid of metadata related to dropped collections
process.out.outputCommands = [ 'drop *' ]

# Explicit list of collections to keep (basis is default PAT event content)
process.out.outputCommands.extend( [ # PAT Objects
                                     'keep *_cleanLayer1Photons_*_*',
                                     'keep *_cleanLayer1Electrons_*_*',
                                     'keep *_cleanLayer1Muons_*_*',
                                     'keep *_cleanLayer1Taus_*_*',
                                     'keep *_cleanLayer1Jets*_*_*',       # All Jets
                                     'keep *_layer1METs*_*_*',            # All METs
                                     'keep *_cleanLayer1Hemispheres_*_*',
                                     'keep *_cleanLayer1PFParticles_*_*',
                                     # Generator information
                                     'keep GenEventInfoProduct_generator_*_*',
                                     'keep GenRunInfoProduct_generator_*_*',
                                     # Generator particles/jets/MET
                                     'keep recoGenParticles_genParticles_*_*',
                                     'keep recoGenJets_iterativeCone5GenJets_*_*',
                                     'keep recoGenJets_sisCone5GenJets_*_*',
                                     'keep recoGenJets_antikt5GenJets_*_*',
                                     'keep recoGenMETs_*_*_*',
                                     # Trigger information
                                     'keep edmTriggerResults_TriggerResults_*_HLT',
                                     'keep *_hltTriggerSummaryAOD_*_*',
                                     'keep L1GlobalTriggerObjectMapRecord_*_*_*',
                                     'keep L1GlobalTriggerReadoutRecord_*_*_*',
                                     # Others
                                     'keep *_muon*METValueMapProducer_*_*',   # Muon corrections to MET
                                     'keep *_offlinePrimaryVertices_*_*',
                                     'keep *_offlineBeamSpot_*_*',
                                     'keep *_towerMaker_*_*',                 # Keep CaloTowers for cross-cleaning
                                     'keep recoTracks_generalTracks_*_*',
                                     'keep recoSuperClusters_corrected*_*_*',
                                     'keep recoConversions_conversions_*_*',
                                     'keep recoTracks_*onversions_*_*',
                                     'keep HcalNoiseSummary_*_*_'+process.name_(), # Only keep the one we create
                                     'keep recoPFCandidates_particleFlow_*_*'
                                     ] )


#-- Trigger matching ----------------------------------------------------------
from PhysicsTools.PatAlgos.tools.trigTools import *
switchOnTrigger( process )
process.patTriggerSequence.remove( process.patTriggerMatcher )
process.patTriggerEvent.patTriggerMatches  = ()


#-- Execution path ------------------------------------------------------------
# Rename default jet collection for uniformity
process.cleanLayer1JetsIC5 = process.cleanLayer1Jets
process.layer1METsIC5      = process.layer1METs

# Modify subsequent modules
process.cleanLayer1Hemispheres.patJets = process.cleanLayer1JetsIC5.label()
process.countLayer1Jets.src            = process.cleanLayer1JetsIC5.label()

# Modify counters' input
process.allLayer1Summary.candidates.remove(cms.InputTag('layer1METs'))
process.allLayer1Summary.candidates.append(cms.InputTag('layer1METsIC5'))
process.cleanLayer1Summary.candidates.remove(cms.InputTag('cleanLayer1Jets'))
process.cleanLayer1Summary.candidates.append(cms.InputTag('cleanLayer1JetsIC5'))
# Add new jet collections to counters (MET done automatically)
for jets in ( 'AK5', 'SC5','IC5PF','SC5Track', 'AK5JPT', 'SC5JPT' ):
    process.allLayer1Summary.candidates.append(cms.InputTag('allLayer1Jets'+jets))
    process.selectedLayer1Summary.candidates.append(cms.InputTag('selectedLayer1Jets'+jets))
    process.cleanLayer1Summary.candidates.append(cms.InputTag('cleanLayer1Jets'+jets))

# Full path
process.p = cms.Path( process.hcalnoise*process.addTrackJets*process.JetPlusTrackCorrections
                      * process.patDefaultSequence
                      * process.patTrigger*process.patTriggerEvent )

