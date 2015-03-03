import FWCore.ParameterSet.Config as cms

#define you jet ID
jetID = cms.InputTag("ak4PFJetsCHS")

#JTA for your jets
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
myak4JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
                                                  j2tParametersVX,
                                                  jets = jetID
                                                  )

#new input for impactParameterTagInfos, softleptons
from RecoBTag.Configuration.RecoBTag_cff import *
impactParameterTagInfos.jetTracks = cms.InputTag("myak4JetTracksAssociatorAtVertex")
pfImpactParameterTagInfos.jets = jetID
softPFMuonsTagInfos.jets = jetID
softPFElectronsTagInfos.jets = jetID

#for MC do the matching with you jet collection
from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons
from PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi import ak4JetFlavourInfos
ak4JetFlavourInfos.jets = jetID

from PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi import *
AK4byRef.jets = jetID

#do the matching : only for MC
flavourSeq = cms.Sequence(
    selectedHadronsAndPartons *
    ak4JetFlavourInfos
    )

oldFlavourSeq = cms.Sequence(
    myPartons *
    AK4Flavour
    )

#run the btagging sequence for your jets
btagSequence = cms.Sequence(
    myak4JetTracksAssociatorAtVertex *
    btagging
    )

#select good primary vertex
from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices

#to select events passing a scpecific trigger
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
bTagHLT  = hltHighLevel.clone(TriggerResultsTag = "TriggerResults::HLT", HLTPaths = ["HLT_PFJet40_v*"])

