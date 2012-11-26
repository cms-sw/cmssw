import FWCore.ParameterSet.Config as cms


def quadrereco(process):
   
    process.load("FWCore.MessageService.MessageLogger_cfi")
    process.load('Configuration.StandardSequences.Services_cff')
    process.load('Configuration.StandardSequences.DigiToRaw_cff')
    process.load('Configuration.StandardSequences.RawToDigi_cff')
    process.load('Configuration.Geometry.GeometryIdeal_cff')
    process.load('Configuration.StandardSequences.MagneticField_38T_cff')        
    process.load('Configuration.StandardSequences.Reconstruction_cff')
   
    process.Dump = cms.EDAnalyzer("EventContentAnalyzer")
    
    process.quadrereco = cms.Sequence(
               #localReco
    	       process.siPixelRecHits*
    	       process.siStripMatchedRecHits*
    	       #process.ecalLocalRecoSequence*
    	       
               #globalReco
    	       process.offlineBeamSpot*
               process.recopixelvertexing*
               process.trackingGlobalReco*
               process.caloTowersRec*
               process.vertexreco*
               process.egammaGlobalReco
               #process.Dump
               )
    
    return process