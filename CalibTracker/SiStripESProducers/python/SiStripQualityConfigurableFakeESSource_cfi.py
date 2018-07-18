import FWCore.ParameterSet.Config as cms

siStripQualityConfigurableFakeESSource = cms.ESSource("SiStripQualityConfigurableFakeESSource",
    printDebug = cms.untracked.bool(False),
    appendToDataLabel = cms.string(''),
    BadComponentList = cms.untracked.VPSet(cms.PSet(
        SubDet = cms.string('TIB'),  
        layer = cms.uint32(0),        ## SELECTION: layer = 1..4, 0(ALL)		    
        bkw_frw = cms.uint32(0),      ## bkw_frw = 1(TIB-), 2(TIB+) 0(ALL)	    
        detid = cms.uint32(0),        ## int_ext = 1 (internal), 2(external), 0(ALL)  
        ster = cms.uint32(0),         ## ster = 1(stereo), 2 (nonstereo), 0(ALL)	    
        string_ = cms.uint32(0),      ## string = 1..N, 0(ALL)			    
        int_ext = cms.uint32(0)       ## detid number = 0 (ALL),  specific number     
    ), 
        cms.PSet(
            SubDet = cms.string('TID'), 
            wheel = cms.uint32(0),      ## SELECTION: side = 1(back, Z-), 2(front, Z+), 0(ALL)	 
            detid = cms.uint32(0),      ## wheel = 1..3, 0(ALL)					 
            ster = cms.uint32(0),       ## ring  = 1..3, 0(ALL)					 
            ring = cms.uint32(0),       ## ster = 1(stereo), 2 (nonstereo), 0(ALL)		 
            side = cms.uint32(0)            ## detid number = 0 (ALL),  specific number           
        ), 
        cms.PSet(
            SubDet = cms.string('TOB'),
            layer = cms.uint32(3),    ## SELECTION: layer = 1..6, 0(ALL)	       
            bkw_frw = cms.uint32(0),  ## bkw_frw = 1(TOB-) 2(TOB+) 0(everything)     
            rod = cms.uint32(0),      ## rod = 1..N, 0(ALL)			       
            detid = cms.uint32(0),       ## ster = 1(stereo), 2 (nonstereo), 0(ALL)  
            ster = cms.uint32(0)         ## detid number = 0 (ALL),  specific number 
        ), 
        cms.PSet(
            SubDet = cms.string('TEC'),
            wheel = cms.uint32(0),             ## SELECTION: side = 1(TEC-), 2(TEC+),  0(ALL)	
            petal = cms.uint32(0),             ## wheel = 1..9, 0(ALL)				
            detid = cms.uint32(0),             ## petal_bkw_frw = 1(backward) 2(forward) 0(all)	
            ster = cms.uint32(0),              ## petal = 1..8, 0(ALL)				
            petal_bkw_frw = cms.uint32(0),     ## ring = 1..7, 0(ALL)				
            ring = cms.uint32(0),              ## ster = 1(sterero), else(nonstereo), 0(ALL)	
            side = cms.uint32(0)                  ## detid number = 0 (ALL),  specific number    
        ))
)


