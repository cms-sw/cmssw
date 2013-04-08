import FWCore.ParameterSet.Config as cms

process = cms.Process("OWNPARTICLES")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/user/geisler/RelValZmumuJets_Pt_20_300_PU_START53_V6-v1_GEN-SIM-RECO.root'),
    secondaryFileNames =  cms.untracked.vstring('file:/user/geisler/RelValZmumuJets_Pt_20_300_PU_START53_V6-v1_GEN-SIM-DIGI-RAW-HLTDEBUG.root'),
)

#process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('file:/user/geisler/QCD_Pt-15to3000_Tune2C_Flat_8TeV_pythia8_AODSIM.root'),
#)

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START53_V11::All'
		
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
		
### AssociationMap-specific includes
from CommonTools.RecoUtils.pf_pu_assomap_cfi import AssociationMaps
		
process.assMap = AssociationMaps.clone(
          VertexCollection = cms.InputTag('selectedPrimaryVertexQuality'),
)				  

### FirstVertexTracks-specific includes
from CommonTools.RecoUtils.pf_pu_firstvertextracks_cfi import FirstVertexTracks
						       
process.firstVertexTracks = FirstVertexTracks.clone(
          AssociationMap = cms.InputTag('assMap'),
          VertexCollection = cms.InputTag('selectedPrimaryVertexQuality'),
)
		

  
process.p = cms.Path(
	process.selectedPrimaryVertexQuality *
	process.assMap *
	process.firstVertexTracks	
)

process.output = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string("myOutput.root"),
 	outputCommands = cms.untracked.vstring('drop *',
		  'keep *_*_*_RECO'),
)

process.out_step = cms.EndPath(process.output)
		
