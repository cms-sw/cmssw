import FWCore.ParameterSet.Config as cms

process = cms.Process("PFJETS")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/user/geisler/QCD_Pt-15to3000_Tune2C_Flat_8TeV_pythia8_AODSIM.root'),
)

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START53_V11::All'
		
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

### standard includes
process.load('Configuration.Geometry.GeometryPilot2_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
		
	
process.selectedPrimaryVertexQuality = cms.EDFilter("VertexSelector",
   	src = cms.InputTag('offlinePrimaryVertices'),
	cut = cms.string("isValid & ndof >= 4 & chi2 > 0 & tracksSize > 0 & abs(z) < 24 & abs(position.Rho) < 2."),
	filter = cms.bool(False),
)
		
### PFCandidate AssociationMap-specific includes
from CommonTools.RecoUtils.pfcand_assomap_cfi import PFCandAssoMap
		
process.PFCand2VertexAM = PFCandAssoMap.clone(
          VertexCollection = cms.InputTag('selectedPrimaryVertexQuality'),
)
		
### PFCandidateCollection-specific includes
from CommonTools.RecoUtils.pfcand_nopu_witham_cfi import FirstVertexPFCandidates
		
process.PFCand = FirstVertexPFCandidates.clone(
          VertexPFCandAssociationMap = cms.InputTag('PFCand2VertexAM'),
          VertexCollection = cms.InputTag('selectedPrimaryVertexQuality'),
)
	
### JetProducer-specific includes
from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets	

process.ak5PFJetsNew = ak5PFJets.clone(
	src = cms.InputTag("PFCand","P2V")
	#src = cms.InputTag("PFCand","V2P")
)

process.load("JetMETCorrections.Configuration.JetCorrectionServices_cff")
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")

# L2L3 Correction Producers
process.ak5PFJetsNewL23 = cms.EDProducer('PFJetCorrectionProducer',
    src        = cms.InputTag('ak5PFJetsNew'),
    correctors = cms.vstring('ak5PFL2L3')
)
		
# L1L2L3 Correction Producers
process.ak5PFJetsNewL123 = cms.EDProducer('PFJetCorrectionProducer',
    src        = cms.InputTag('ak5PFJetsNew'),
    correctors = cms.vstring('ak5PFL1L2L3')
)
				
### paths & sequences
		
##sequence to produce the collection of pfcand's associated to the first vertex
process.pfc = cms.Sequence(
	  process.selectedPrimaryVertexQuality
	* process.PFCand2VertexAM
	* process.PFCand
)
		
##sequence to produce the jet collections
process.pfjet = cms.Sequence(
	  process.ak5PFJetsNew
	* process.ak5PFJetsNewL23
	* process.ak5PFJetsNewL123
)
		

  
process.p = cms.Path( 
	  process.pfc
	* process.pfjet
)
		
process.myOutput = cms.OutputModule("PoolOutputModule",
     	fileName = cms.untracked.string('myOutput.root'),
 	outputCommands = cms.untracked.vstring('drop *',
		  'keep *_*_*_PFJETS'),
)
  
process.e = cms.EndPath( process.myOutput )
