import FWCore.ParameterSet.Config as cms

# PAT Layer 0+1
from PhysicsTools.PatAlgos.patLayer0_cff import *
from PhysicsTools.PatAlgos.recoLayer0.tauDiscriminators_cff import *
from PhysicsTools.PatAlgos.patLayer1_cff import *
from PhysicsTools.PatAlgos.producersLayer1.pfParticleProducer_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import * 
from PhysicsTools.PatAlgos.cleaningLayer0.pfJetCleaner_cfi import allLayer0PFJets

myJets = cms.InputTag("pfTopProjection:PFJets")
allLayer0PFJets.jetSource = myJets
allLayer0PFJets.removeOverlaps = cms.PSet(
    electrons = cms.PSet( 
        collection = cms.InputTag("allLayer0Electrons"), 
        deltaR = cms.double(0.3),
        cut = cms.string('pt > 10'),
        flags = cms.vstring('Isolation/All'), #request the item to be marked as isolated
                                              #by the PATElectronCleaner
    )
)
allLayer0PFJets.bitsToIgnore = cms.vstring('')
myL0Jets = cms.InputTag("allLayer0PFJets")

enableTrigMatch = False

allLayer1Muons.addTrigMatch =  enableTrigMatch
allLayer1Electrons.addTrigMatch =  enableTrigMatch 
allLayer1Jets.addTrigMatch =  enableTrigMatch
allLayer1Taus.addTrigMatch =  enableTrigMatch
allLayer1METs.addTrigMatch =  enableTrigMatch



jetGenJetMatch.src = myL0Jets
jetPartonAssociation.jets = myL0Jets
jetPartonMatch.src        = myL0Jets

jetPartonMatch.mcStatus = cms.vint32(2)
jetPartonMatch.mcPdgId  = cms.vint32(1, 2, 3, 4, 5, 11, 13, 15, 21)

# Trigger match
#TODO check the trigger match
jetTrigMatchHLT1ElectronRelaxed.src = myL0Jets
jetTrigMatchHLT2jet.src             = myL0Jets

jtAssoc = cms.EDFilter(
    "JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = myJets
)
patAODJetTracksAssociator.src       = myJets
patAODJetTracksAssociator.tracks    = cms.InputTag("jtAssoc")

AODJetCharge.src = myJets
# Layer 1 pat::Jets
allLayer1Jets.jetSource = myL0Jets

#TODO: make the following input optional?
#depends on the association procedure
layer0JetTracksAssociator.collection = myL0Jets
layer0JetTracksAssociator.backrefs = myL0Jets
layer0JetCharge.src = myL0Jets

# no calo towers in particle flow jet
allLayer1Jets.embedCaloTowers   = False
allLayer1Jets.addBTagInfo       = False
allLayer1Jets.addResolutions    = False

#TODO how to deal with the jet corrections ?
allLayer1Jets.addJetCorrFactors = False

# jet corrections 
#jetCorrFactors.jetSource = cms.InputTag("topProjection")

allLayer1Jets.jetSource = myL0Jets
# replaces for MET ---------------------------------------------------

# allLayer1METs.src does not crash !!
myMET = "pfMET"
allLayer1METs.metSource = cms.InputTag( myMET )
metTrigMatchHLT1MET65.src = cms.InputTag( myMET )

# replaces for Taus --------------------------------------------------

#taus = "pfTaus"

#allLayer1Taus.tauSource = cms.InputTag( all )
#tauMatch.src = cms.InputTag( taus )
#tauGenJetMatch.src = cms.InputTag( taus )
#tauTrigMatchHLT1Tau.src = cms.InputTag( taus )

tauIDSources = cms.PSet(
    byIsolation = cms.InputTag("patPFRecoTauDiscriminationByIsolation"),
    againstElectron = cms.InputTag("patPFRecoTauDiscriminationAgainstElectron"),
    againstMuon = cms.InputTag("patPFRecoTauDiscriminationAgainstMuon")
)



# replaces for Muons -------------------------------------------------

muons = "pfMuons"

allLayer1Muons.pfMuonSource =  cms.InputTag( muons )
allLayer1Muons.useParticleFlow =  cms.bool( True )
muonMatch.src = cms.InputTag( muons )
allLayer1Muons.addGenMatch = True
allLayer1Muons.embedPFCandidate = True

muonTrigMatchHLT1MuonNonIso.src = cms.InputTag( muons )
muonTrigMatchHLT1MET65.src =  cms.InputTag( muons )



# Stuff to incorporate e-gamma electrons (w/ std PAT cfgs) -----------
# Confing for layer0 electrons already included,
# but isolation should not be ignared 
allLayer0Electrons.bitsToIgnore = cms.vstring('')



# simplifying the PAT sequences --------------------------------------

# if we forget , no warning..
#load("PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff")
#from PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff import *



jetTrackAssociation =  cms.Sequence(
    jtAssoc +
    patAODJetTracksAssociator 
    + layer0JetTracksAssociator + layer0JetCharge
)

patHighLevelReco = cms.Sequence(
    patJetFlavourId +
    patAODJetTracksCharge 
)

patMCTruth_withoutElectronPhoton = cms.Sequence(
    patMCTruth_withoutLeptonPhoton +
    muonMatch
    )

patLayer1 = cms.Sequence(
    layer1Jets +
    allLayer1PFParticles + 
    layer1METs +
    patPFTauDiscrimination +
    layer1Taus +
    layer1Muons
)

# disabling trigger matching, due to a dictionnary inconsistency
# between 2_1_X and 2_2_X


patTrigMatch = cms.Sequence(
    #    patTrigMatchCandHLT1ElectronStartup +
    #    patTrigMatchHLT1PhotonRelaxed +
    #    patTrigMatchHLT1ElectronRelaxed +
    patHLT1ElectronRelaxed +
    patHLT1Tau + 
    jetTrigMatchHLT1ElectronRelaxed +
    patTrigMatchHLT1MuonNonIso +
    patTrigMatchHLT2jet +
    patTrigMatchHLT1MET65
    + tauTrigMatchHLT1Tau
)

patFromPF2PAT = cms.Sequence (
    #stuff for e-gamma electrons before level0
    patAODElectronIsolation + 
    #layer-0 electrons
    allLayer0Electrons + 
#    allLayer0Electrons +
#    allLayer0Potons + 
#    patTrigMatch +
    #layer-0 jets after topProjection and PAT-electron cleanings 
    allLayer0PFJets + 
    #stuff for e-gamma electrons after level0
    patElectronId + 
    patLayer0ElectronIsolation + 
    jetTrackAssociation +
    patHighLevelReco +
    patMCTruth_withoutElectronPhoton +
    #stuff for e-gamma electrons gen matching 
    electronMatch+ 
    patLayer1
    #stuff for e-gamma electrons at level1 
    + layer1Electrons 
)
