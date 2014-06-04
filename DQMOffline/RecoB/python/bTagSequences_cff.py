import FWCore.ParameterSet.Config as cms

#define you jet ID
jetID = cms.InputTag("ak5PFJetsCHS")
corr = 'ak5PFCHSL1FastL2L3'
#JTA for your jets
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
myak5JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
                                                  j2tParametersVX,
                                                  jets = jetID
                                                  )

#new input for impactParameterTagInfos, softleptons
from RecoBTag.Configuration.RecoBTag_cff import *
impactParameterTagInfos.jetTracks = cms.InputTag("myak5JetTracksAssociatorAtVertex")
softPFMuonsTagInfos.jets = jetID
softPFElectronsTagInfos.jets = jetID

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
from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices

#to select events passing a scpecific trigger
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
bTagHLT  = hltHighLevel.clone(TriggerResultsTag = "TriggerResults::HLT", HLTPaths = ["HLT_PFJet40_v*"])

#for the  use of JEC, could change with time : be careful if recommandations change for the correctors
#define you sequence like  process.JECAlgo = cms.Sequence(process.ak5PFJetsJEC * process.PFJetsFilter)
JetCut=cms.string("neutralHadronEnergyFraction < 0.99 && neutralEmEnergyFraction < 0.99 && nConstituents > 1 && chargedHadronEnergyFraction > 0.0 && chargedMultiplicity > 0.0 && chargedEmEnergyFraction < 0.99")
#JetCut=cms.string("chargedEmEnergyFraction < 99999")

from JetMETCorrections.Configuration.DefaultJEC_cff import *
from JetMETCorrections.Configuration.JetCorrectionServices_cff import *
ak5PFchsL1Fastjet = ak5PFL1Fastjet.clone(algorithm='AK5PFchs', srcRho=cms.InputTag('kt6PFJets','rho'))
ak5PFchsL2Relative = ak5PFL2Relative.clone( algorithm = 'AK5PFchs' )
ak5PFchsL3Absolute = ak5PFL3Absolute.clone( algorithm = 'AK5PFchs' )
ak5PFchsResidual = ak5PFResidual.clone( algorithm = 'AK5PFchs' )
ak5PFCHSL2L3 = cms.ESProducer(
        'JetCorrectionESChain',
            correctors = cms.vstring('ak5PFchsL2Relative','ak5PFchsL3Absolute')
            )
ak5PFCHSL2L3Residual = cms.ESProducer(
        'JetCorrectionESChain',
            correctors = cms.vstring('ak5PFchsL2Relative','ak5PFchsL3Absolute','ak5PFchsResidual')
            )
ak5PFCHSL1FastL2L3 = cms.ESProducer(
        'JetCorrectionESChain',
            correctors = cms.vstring('ak5PFchsL1Fastjet','ak5PFchsL2Relative','ak5PFchsL3Absolute')
            )
ak5PFCHSL1FastL2L3Residual = cms.ESProducer(
        'JetCorrectionESChain',
            correctors = cms.vstring('ak5PFchsL1Fastjet','ak5PFchsL2Relative','ak5PFchsL3Absolute','ak5PFchsResidual')
            )

ak5JetsJEC = ak5PFJetsL2L3.clone(
        src = jetID,
        correctors = [corr]        )

PFJetsFilter = cms.EDFilter("PFJetSelector",
                            src = cms.InputTag("ak5JetsJEC"),
                            cut = JetCut,
                            filter = cms.bool(True)
                            )
