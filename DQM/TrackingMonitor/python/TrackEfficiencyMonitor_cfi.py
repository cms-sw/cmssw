# The following comments couldn't be translated into the new config version:

# All/OuterSurface/InnerSurface/ImpactPoint/default(track)
#

import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
TrackEffMon = DQMEDAnalyzer('TrackEfficiencyMonitor',
    theRadius = cms.double(85.0),
    theMaxZ = cms.double(110.0),
    isBFieldOff = cms.bool(False),
    TKTrackCollection = cms.InputTag("rsWithMaterialTracksP5"),
    STATrackCollection = cms.InputTag("cosmicMuons"),
    trackEfficiency  = cms.bool(True),    
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName = cms.string('MonitorTrackEfficiency.root'),

    FolderName = cms.string('Track/Efficiencies'),
    AlgoName = cms.string('CTF'),
    muoncoll = cms.InputTag('muons'),        
    muonXBin =  cms.int32(50),
    muonXMin = cms.double(-100),
    muonXMax = cms.double(100),
     
    muonYBin =  cms.int32(50),
    muonYMin = cms.double(-100),
    muonYMax = cms.double(100),
    
    muonZBin =  cms.int32(50),
    muonZMin = cms.double(-500),
    muonZMax = cms.double(500),
 
    muonEtaBin =  cms.int32(50),
    muonEtaMin = cms.double(-3.2),
    muonEtaMax = cms.double(3.2),
    
    muonPhiBin =  cms.int32(50),
    muonPhiMin = cms.double(-3.2),
    muonPhiMax = cms.double(0.),
    
    muonD0Bin =  cms.int32(50),
    muonD0Min = cms.double(-100),
    muonD0Max = cms.double(100),
  
    muonCompatibleLayersBin =  cms.int32(10),
    muonCompatibleLayersMin = cms.double(0),
    muonCompatibleLayersMax = cms.double(30),
    
    trackXBin =  cms.int32(50),
    trackXMin = cms.double(-100),
    trackXMax = cms.double(100),
     
    trackYBin =  cms.int32(50),
    trackYMin = cms.double(-100),
    trackYMax = cms.double(100),
 
    trackZBin =  cms.int32(50),
    trackZMin = cms.double(-500),
    trackZMax = cms.double(500),
    
    trackEtaBin =  cms.int32(50),
    trackEtaMin = cms.double(-3.2),
    trackEtaMax = cms.double(3.2),
    
    trackPhiBin =  cms.int32(50),
    trackPhiMin = cms.double(-3.2),
    trackPhiMax = cms.double(0.),
    
    trackD0Bin =  cms.int32(50),
    trackD0Min = cms.double(-100),
    trackD0Max = cms.double(100),
     
    trackCompatibleLayersBin =  cms.int32(10),
    trackCompatibleLayersMin = cms.double(0),
    trackCompatibleLayersMax = cms.double(30),
    
    deltaXBin = cms.int32(50),
    deltaXMin = cms.double(-100),
    deltaXMax = cms.double(100),
    
    deltaYBin = cms.int32(50),
    deltaYMin = cms.double(-100),
    deltaYMax = cms.double(100),
    
    signDeltaXBin = cms.int32(50),
    signDeltaXMin = cms.double(-5),
    signDeltaXMax = cms.double(5),
    
    signDeltaYBin = cms.int32(50),
    signDeltaYMin = cms.double(-5),
    signDeltaYMax = cms.double(5),
 
)   
    
    
