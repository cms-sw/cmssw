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
softPFMuonsTagInfos.jets = jetID
softPFElectronsTagInfos.jets = jetID

#for MC do the matching with you jet collection
from PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi import *
AK4byRef.jets = jetID

#do the matching : only for MC
flavourSeq = cms.Sequence(
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

#for the  use of JEC, could change with time : be careful if recommandations change for the correctors
#define you sequence like  process.JECAlgo = cms.Sequence(process.ak4PFJetsJEC * process.PFJetsFilter)
JetCut=cms.string("neutralHadronEnergyFraction < 0.99 && neutralEmEnergyFraction < 0.99 && nConstituents > 1 && chargedHadronEnergyFraction > 0.0 && chargedMultiplicity > 0.0 && chargedEmEnergyFraction < 0.99")
#JetCut=cms.string("chargedEmEnergyFraction < 99999")

from JetMETCorrections.Configuration.DefaultJEC_cff import *
from JetMETCorrections.Configuration.JetCorrectionServices_cff import *


PFJetsFilter = cms.EDFilter("PFJetSelector",
                            src = cms.InputTag("ak4PFJetsL2L3"),
                            cut = JetCut,
                            filter = cms.bool(True)
                            )
