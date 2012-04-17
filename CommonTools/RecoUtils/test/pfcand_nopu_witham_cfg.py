import FWCore.ParameterSet.Config as cms

process = cms.Process("PFCAND")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/user/geisler/Test.root')
)
		
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START52_V9::All'

### standard includes
process.load('Configuration.StandardSequences.GeometryPilot2_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
	
process.selectedPrimaryVertexQuality = cms.EDFilter("VertexSelector",
   	src = cms.InputTag('offlinePrimaryVertices'),
	cut = cms.string("isValid & ndof >= 4 & chi2 > 0 & tracksSize > 0 & abs(z) < 24 & abs(position.Rho) < 2."),
	filter = cms.bool(False),
)

### AssociationMap-specific includes		
from CommonTools.RecoUtils.pf_pu_assomap_cfi import Tracks2Vertex
		
process.Tracks2VertexAM = Tracks2Vertex.clone(
        VertexCollection = cms.InputTag('selectedPrimaryVertexQuality'),
	VertexAssOneDim = cms.untracked.bool(False),
	VertexAssUseAbsDistance = cms.untracked.bool(True),
)
		
### PFCandidateCollection-specific includes
from CommonTools.RecoUtils.pfcand_nopu_witham_cfi import FirstVertexPFCandidates
		
process.PFCand = FirstVertexPFCandidates.clone(
          VertexCollection = cms.InputTag('selectedPrimaryVertexQuality'),
          VertexTrackAssociationMap = cms.InputTag('Tracks2VertexAM'),
)

  
process.p = cms.Path(  
	  process.selectedPrimaryVertexQuality
	* process.Tracks2VertexAM
	* process.PFCand
)
		
process.myOutput = cms.OutputModule("PoolOutputModule",
     	fileName = cms.untracked.string('myOutput.root')
)
  
process.e = cms.EndPath( process.myOutput )
