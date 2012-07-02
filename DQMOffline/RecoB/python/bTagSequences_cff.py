import FWCore.ParameterSet.Config as cms

#define you jet ID
jetID = cms.InputTag("ak5PFJets")

#JTA for your jets
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
myak5JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
                                                  j2tParametersVX,
                                                  jets = jetID
                                                  )

#new input for impactParameterTagInfos, softleptons
from RecoBTag.Configuration.RecoBTag_cff import *
impactParameterTagInfos.jetTracks = cms.InputTag("myak5JetTracksAssociatorAtVertex")
softMuonTagInfos.jets = jetID
softElectronTagInfos.jets = jetID

#for MC do the matching with you jet collection
from PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi import *
AK5byRef.jets = jetID

#do the matching : only for MC
flavourSeq = cms.Sequence(
    myPartons *
    AK5Flavour
    )

#run the btagging sequence for your jets
btagSequence = cms.Sequence(
    myak5JetTracksAssociatorAtVertex *
    btagging
    )   

#select good primary vertex
from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector
goodOfflinePrimaryVertices = cms.EDFilter(
    "PrimaryVertexObjectFilter",
    filterParams = pvSelector.clone( minNdof = cms.double(4.0), maxZ = cms.double(24.0) ),
    src=cms.InputTag('offlinePrimaryVertices')
    )

#to select events passing a scpecific trigger
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
bTagHLT  = hltHighLevel.clone(TriggerResultsTag = "TriggerResults::HLT", HLTPaths = ["HLT_PFJet40_v*"])
