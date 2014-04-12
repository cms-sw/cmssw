# The following comments couldn't be translated into the new config version:

# All/OuterSurface/InnerSurface/ImpactPoint/default(eff)
#

import FWCore.ParameterSet.Config as cms

TrackEffClient = cms.EDAnalyzer("TrackEfficiencyClient",
  
    FolderName = cms.string('Track/Efficiencies'),
    AlgoName = cms.string('CTF'),
    trackEfficiency = cms.bool(True),
        
    effXBin =  cms.int32(50),
    effXMin = cms.double(-100),
    effXMax = cms.double(100),
     
    effYBin =  cms.int32(50),
    effYMin = cms.double(-100),
    effYMax = cms.double(100),
 
    effZBin =  cms.int32(50),
    effZMin = cms.double(-500),
    effZMax = cms.double(500),
    
    effEtaBin =  cms.int32(50),
    effEtaMin = cms.double(-3.2),
    effEtaMax = cms.double(3.2),
    
    effPhiBin =  cms.int32(50),
    effPhiMin = cms.double(-3.2),
    effPhiMax = cms.double(0.),
    
    effD0Bin =  cms.int32(50),
    effD0Min = cms.double(-100),
    effD0Max = cms.double(100),
     
    effCompatibleLayersBin =  cms.int32(10),
    effCompatibleLayersMin = cms.double(0),
    effCompatibleLayersMax = cms.double(30),
   
)   
    
    
