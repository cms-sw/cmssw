import FWCore.ParameterSet.Config as cms
import copy 

# -----------------------------------------------------------------------
# Default configuration

default = cms.VPSet(
    #### Barrel Pixel HB X- 
    cms.PSet(alignableId       = cms.string("TPBHalfBarrelXminus"),
             DOF               = cms.string("X"),
             cut               = cms.double(5.0),
             sigCut            = cms.double(2.5),
             maxMoveCut        = cms.double(200.0),
             maxErrorCut       = cms.double(10.0)
             ),
        
    cms.PSet(alignableId        = cms.string("TPBHalfBarrelXminus"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPBHalfBarrelXminus"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),                       
    
    cms.PSet(alignableId        = cms.string("TPBHalfBarrelXminus"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPBHalfBarrelXminus"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPBHalfBarrelXminus"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)                         
             ),
    
    ### Barrel Pixel HB X+
    cms.PSet(alignableId        = cms.string("TPBHalfBarrelXplus"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPBHalfBarrelXplus"),
             DOF                = cms.string("thetaX"),        
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0),
             ),
    
    cms.PSet(alignableId        = cms.string("TPBHalfBarrelXplus"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),                       
                 
    cms.PSet(alignableId        = cms.string("TPBHalfBarrelXplus"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPBHalfBarrelXplus"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPBHalfBarrelXplus"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)                         
             ),
    
    ### Forward Pixel HC X-,Z-
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXminusZminus"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXminusZminus"),
             DOF                = cms.string("thetaX"),                                                                       
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXminusZminus"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),                       
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXminusZminus"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXminusZminus"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXminusZminus"),
             DOF                = cms.string("thetaZ"),         
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)                         
             ),
    
    ### Forward Pixel HC X+,Z-
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXplusZminus"),
             DOF                = cms.string("X"),         
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXplusZminus"),
             DOF                = cms.string("thetaX"),                  
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXplusZminus"),
             DOF                = cms.string("Y"),        
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),                       
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXplusZminus"),
             DOF                = cms.string("thetaY"),                 
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXplusZminus"),
             DOF                = cms.string("Z"),         
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXplusZminus"),
             DOF                = cms.string("thetaZ"),         
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)                         
             ),
    
    ### Forward Pixel HC X-,Z+
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXminusZplus"),
             DOF                = cms.string("X"),        
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut       =  cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXminusZplus"),
             DOF                = cms.string("thetaX"),                 
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXminusZplus"),
             DOF                = cms.string("Y"),        
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXminusZplus"),
             DOF                = cms.string("thetaY"),                
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXminusZplus"),
             DOF                = cms.string("Z"),                
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXminusZplus"),
             DOF                = cms.string("thetaZ"),        
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)                         
             ),
    
    ### Forward Pixel HC X+,Z+
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXplusZplus"),
             DOF                = cms.string("X"),       
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),                                                               
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXplusZplus"),
             DOF                = cms.string("thetaX"),       
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXplusZplus"),
             DOF                = cms.string("Y"),       
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXplusZplus"),
             DOF                = cms.string("thetaY"),       
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXplusZplus"),
             DOF                = cms.string("Z"),       
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)
             ),
    
    cms.PSet(alignableId        = cms.string("TPEHalfCylinderXplusZplus"),
             DOF                = cms.string("thetaZ"),                
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10.0)                         
             )
    )

