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
    
#### Include QuadSeed Sequence after SingleLeg Sequence,  before the track merger:::     
    process.ConvStep.insert(5,process.Conv2Step)

#### Merge SingleLeg/QuadSeed tracks:::
    process.conversionStepTracks.TrackProducers = cms.VInputTag(cms.InputTag('convStepTracks'),cms.InputTag('conv2StepTracks'))
    process.conversionStepTracks.hasSelector=cms.vint32(1,1)
    process.conversionStepTracks.selectedTrackQuals = cms.VInputTag(cms.InputTag("convStepSelector","convStep"),cms.InputTag("conv2StepSelector","conv2Step"))
    process.conversionStepTracks.setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) ))
    
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
