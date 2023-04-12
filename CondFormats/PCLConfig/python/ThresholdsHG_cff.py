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
             ),

    ### Barrel Pixel Ladder
    cms.PSet(alignableId        = cms.string("TPBLadder"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadder"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadder"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadder"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(2000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadder"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadder"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    ### EndCap Pixel Panel
    cms.PSet(alignableId        = cms.string("TPEPanel"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanel"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(2000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanel"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanel"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(5000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanel"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(300.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanel"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(1000.0),
             maxErrorCut        = cms.double(10000.0)
             ),
    
    ### Barrel Pixel Ladder Layer 1
    cms.PSet(alignableId        = cms.string("TPBLadderLayer1"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer1"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer1"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer1"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(2000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer1"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer1"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),
    
    ### Barrel Pixel Ladder Layer 2
    cms.PSet(alignableId        = cms.string("TPBLadderLayer2"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer2"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer2"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer2"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(2000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer2"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer2"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),
    
    ### Barrel Pixel Ladder Layer 3
    cms.PSet(alignableId        = cms.string("TPBLadderLayer3"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer3"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer3"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer3"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(2000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer3"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer3"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),
    
    ### Barrel Pixel Ladder Layer 4
    cms.PSet(alignableId        = cms.string("TPBLadderLayer4"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer4"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer4"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer4"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(2000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer4"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPBLadderLayer4"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(150.0),
             maxErrorCut        = cms.double(10000.0)
             ),
    
    ### EndCap Pixel Panel Disk1
    cms.PSet(alignableId        = cms.string("TPEPanelDisk1"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk1"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(2000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk1"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk1"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(5000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk1"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(300.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk1"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(1000.0),
             maxErrorCut        = cms.double(10000.0)
             ),
    
    ### EndCap Pixel Panel Disk2
    cms.PSet(alignableId        = cms.string("TPEPanelDisk2"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk2"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(2000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk2"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk2"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(5000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk2"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(300.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk2"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(1000.0),
             maxErrorCut        = cms.double(10000.0)
             ),
    
    ### EndCap Pixel Panel Disk3
    cms.PSet(alignableId        = cms.string("TPEPanelDisk3"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk3"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(2000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk3"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk3"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(5000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk3"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(300.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDisk3"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(1000.0),
             maxErrorCut        = cms.double(10000.0)
             ),
    
    ### EndCap Pixel Panel DiskM1
    cms.PSet(alignableId        = cms.string("TPEPanelDiskM1"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM1"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(2000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM1"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM1"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(5000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM1"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(300.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM1"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(1000.0),
             maxErrorCut        = cms.double(10000.0)
             ),
    
    ### EndCap Pixel Panel DiskM2
    cms.PSet(alignableId        = cms.string("TPEPanelDiskM2"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM2"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(2000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM2"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM2"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(5000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM2"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(300.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM2"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(1000.0),
             maxErrorCut        = cms.double(10000.0)
             ),
    
    ### EndCap Pixel Panel DiskM3
    cms.PSet(alignableId        = cms.string("TPEPanelDiskM3"),
             DOF                = cms.string("X"),
             cut                = cms.double(5.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM3"),
             DOF                = cms.string("thetaX"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(2000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM3"),
             DOF                = cms.string("Y"),
             cut                = cms.double(10.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(200.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM3"),
             DOF                = cms.string("thetaY"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(5000.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM3"),
             DOF                = cms.string("Z"),
             cut                = cms.double(15.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(300.0),
             maxErrorCut        = cms.double(10000.0)
             ),

    cms.PSet(alignableId        = cms.string("TPEPanelDiskM3"),
             DOF                = cms.string("thetaZ"),
             cut                = cms.double(30.0),
             sigCut             = cms.double(2.5),
             fractionCut        = cms.double(0.25),
             maxMoveCut         = cms.double(1000.0),
             maxErrorCut        = cms.double(10000.0)
             )
    )

