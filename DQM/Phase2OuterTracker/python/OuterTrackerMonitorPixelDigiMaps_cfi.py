import FWCore.ParameterSet.Config as cms

OuterTrackerMonitorPixelDigiMaps = cms.EDAnalyzer('OuterTrackerMonitorPixelDigiMaps',
    
    TopFolderName = cms.string('Phase2OuterTracker'),


# PixelDigiMaps barrel y vs x 
# PixelDigiMaps forward/backward endcap y vs x
    TH2PixelDigiMaps_Position = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(120),                      
        xmin = cms.double(-120),
        Nbinsy = cms.int32(960),
        ymax = cms.double(120),                      
        ymin = cms.double(-120)
        ),
	
#PixelDigiMaps barrel y vs x zoomed	
    TH2PixelDigiMaps_Barrel_XY_Zoom = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(60),                      
        xmin = cms.double(30),
        Nbinsy = cms.int32(960),
        ymax = cms.double(15),                      
        ymin = cms.double(-15)
        ),

#PixelDigiMaps #rho vs z
    TH2PixelDigiMaps_RZ = cms.PSet(
        Nbinsx = cms.int32(900),
        xmax = cms.double(300),                      
        xmin = cms.double(-300),
        Nbinsy = cms.int32(900),
        ymax = cms.double(120),                      
        ymin = cms.double(0)
        ),

#PixelDigiMaps Forward Endcap #rho vs. z	
    TH2PixelDigiMaps_Endcap_Fw_RZ_Zoom = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(170),                      
        xmin = cms.double(140),
        Nbinsy = cms.int32(960),
        ymax = cms.double(60),                      
        ymin = cms.double(30)
        ),

#PixelDigiMaps Backward Endcap #rho vs. z	
    TH2PixelDigiMaps_Endcap_Bw_RZ_Zoom = cms.PSet(
        Nbinsx = cms.int32(960),
        xmax = cms.double(-140),                      
        xmin = cms.double(-170),
        Nbinsy = cms.int32(960),
        ymax = cms.double(100),                      
        ymin = cms.double(70)          
        )

)
