import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Btag.hltBtagJetMCTools_cff import *
from HLTriggerOffline.Btag.HltBtagValidation_cff import *

#from Validation.RecoB.bTagAnalysis_cfi import *


#hltBtagPartons = cms.EDProducer("PartonSelector",
#   src = cms.InputTag("genParticles"),
#    withLeptons = cms.bool(False)
#)

#hltBtagJetsbyRef = cms.EDProducer("JetPartonMatcher",
#    jets = cms.InputTag("hltBtagCaloJetL1FastJetCorrected","","HLT"),
#    coneSizeToAssociate = cms.double(0.3),
#    partons = cms.InputTag("hltBtagPartons")
#)

#hltBtagJetsbyValAlgo = cms.EDProducer("JetFlavourIdentifier",
#    srcByReference = cms.InputTag("hltBtagJetsbyRef"),
#    physicsDefinition = cms.bool(False)
#)

#hltBtagJetMCTools = cms.Sequence(hltBtagPartons*hltBtagJetsbyRef*hltBtagJetsbyValAlgo)

#process.hltFastPVPixelVertexFilter = cms.EDFilter( "VertexSelector",
#    filter = cms.bool( True ),
#    src = cms.InputTag( "hltFastPrimaryVertex" ),
#    cut = cms.string( "!isFake && ndof > 0 && abs(z) <= 25 && position.Rho <= 2" )
#)

#denominator trigger
#hltBtagTriggerSelection = cms.EDFilter( "TriggerResultsFilter",
#    triggerConditions = cms.vstring(
#      "HLT_PFJet40*"),
#    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
#    l1tResults = cms.InputTag( "simGtDigis" ),
##    l1tResults = cms.InputTag( "hltGtDigis" ),
#    throw = cms.bool( True )
#)

#hltBtagTriggerSelection = cms.EDFilter( "VertexSelector",
#    filter = cms.bool( True ),
#    src = cms.InputTag( "hltVerticesL3" ),
#    cut = cms.string( "!isFake && ndof > 0 && abs(z) <= 25 && position.Rho <= 2" )
#)

##correct the jet used for the matching
#hltBtagJetsbyRef.jets = cms.InputTag("hltSelector4CentralJetsL1FastJet")
##define HltVertexValidationVertices for the vertex DQM validation

HltVertexValidationVerticesFastSim = HltVertexValidationVertices.clone(
Vertex = cms.VInputTag(
#	cms.InputTag("hltFastPrimaryVertex"), 
#	cms.InputTag("hltFastPVPixelVertices"), 
	cms.InputTag("hltVerticesL3"), 
#	cms.InputTag("hltFastPrimaryVertex"), 
	cms.InputTag("hltFastPVPixelVertices"),
)
)

#put all in a path
hltbtagValidationSequenceFastSim = cms.Sequence(
#hltBtagTriggerSelection +
	hltBtagJetMCTools +
	HltVertexValidationVerticesFastSim +
	hltbTagValidation
)	

