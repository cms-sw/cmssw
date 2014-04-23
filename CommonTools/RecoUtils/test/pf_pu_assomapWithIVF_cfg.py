import FWCore.ParameterSet.Config as cms

process = cms.Process("OWNPARTICLES")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('step2.root'),
)

#process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('file:/user/geisler/QCD_Pt-15to3000_Tune2C_Flat_8TeV_pythia8_AODSIM.root'),
#)

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'POSTLS162_V1::All'
		
### standard includes
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.Geometry.GeometryPilot2_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")	
	
process.selectedPrimaryVertexQuality = cms.EDFilter("VertexSelector",
   	src = cms.InputTag('offlinePrimaryVertices'),
	cut = cms.string("isValid & ndof >= 4 & chi2 > 0 & tracksSize > 0 & abs(z) < 24 & abs(position.Rho) < 2."),
	filter = cms.bool(True),
)
		
### IVF-specific includes
process.load("RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff")
		
### AssociationMap-specific includes
from CommonTools.RecoUtils.pf_pu_assomap_cfi import AssociationMaps
		
process.assMap = AssociationMaps.clone(
          VertexCollection = cms.InputTag('selectedPrimaryVertexQuality'),
          IVFVertexCollection = cms.InputTag('inclusiveMergedVertices'),
)				  

### FirstVertexTracks-specific includes
from CommonTools.RecoUtils.pf_pu_firstvertextracks_cfi import FirstVertexTracks
						       
process.firstVertexTracks = FirstVertexTracks.clone(
          AssociationMap = cms.InputTag('assMap'),
          VertexCollection = cms.InputTag('selectedPrimaryVertexQuality'),
)
		

  
process.p = cms.Path(
	( process.selectedPrimaryVertexQuality +
	process.inclusiveVertexing ) *
	process.assMap *
	process.firstVertexTracks	
)

#process.output = cms.OutputModule("PoolOutputModule",
	#fileName = cms.untracked.string("myOutput.root"),
 	#outputCommands = cms.untracked.vstring('drop *',
		  #'keep *_*_*_OWNPARTICLES'),
#)

#process.out_step = cms.EndPath(process.output)
		
