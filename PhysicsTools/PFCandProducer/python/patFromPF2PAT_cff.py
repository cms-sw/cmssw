import FWCore.ParameterSet.Config as cms

# PAT Layer 0+1
from PhysicsTools.PatAlgos.patLayer0_cff import *
from PhysicsTools.PatAlgos.recoLayer0.tauDiscriminators_cff import *
from PhysicsTools.PatAlgos.patLayer1_cff import *
from PhysicsTools.PatAlgos.producersLayer1.pfParticleProducer_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import * 


myJets = cms.InputTag("pfTopProjection:PFJets")

enableTrigMatch = False

allLayer1Muons.addTrigMatch =  enableTrigMatch
allLayer1Electrons.addTrigMatch =  enableTrigMatch 
allLayer1Jets.addTrigMatch =  enableTrigMatch
allLayer1Taus.addTrigMatch =  enableTrigMatch
allLayer1METs.addTrigMatch =  enableTrigMatch



jetGenJetMatch.src = myJets
jetPartonAssociation.jets = myJets
jetPartonMatch.src        = myJets

jetPartonMatch.mcStatus = cms.vint32(2)
jetPartonMatch.mcPdgId  = cms.vint32(1, 2, 3, 4, 5, 11, 13, 15, 21)

# Trigger match
#TODO check the trigger match
jetTrigMatchHLT1ElectronRelaxed.src = myJets
jetTrigMatchHLT2jet.src             = myJets

jtAssoc = cms.EDFilter(
    "JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = myJets
)
patAODJetTracksAssociator.src       = myJets
patAODJetTracksAssociator.tracks    = cms.InputTag("jtAssoc")

AODJetCharge.src = myJets
# Layer 1 pat::Jets
allLayer1Jets.jetSource = myJets

#TODO: make the following input optional?
#depends on the association procedure
allLayer1Jets.trackAssociationSource = cms.InputTag("patAODJetTracksAssociator")
allLayer1Jets.jetChargeSource = cms.InputTag("AODJetCharge")

# no calo towers in particle flow jet
allLayer1Jets.embedCaloTowers   = False
allLayer1Jets.addBTagInfo       = False
allLayer1Jets.addResolutions    = False

#TODO how to deal with the jet corrections ?
allLayer1Jets.addJetCorrFactors = False

# jet corrections 
#jetCorrFactors.jetSource = cms.InputTag("topProjection")

allLayer1Jets.jetSource = myJets
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

# simplifying the PAT sequences --------------------------------------

# if we forget , no warning..
#load("PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff")
#from PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff import *



jetTrackAssociation =  cms.Sequence(
    jtAssoc +
    patAODJetTracksAssociator 
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
#    allLayer0Electrons +
#    allLayer0Potons + 
#    patTrigMatch +
    jetTrackAssociation +
    patHighLevelReco +
    patMCTruth_withoutElectronPhoton +
    patLayer1
)
