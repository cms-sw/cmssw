#ifndef Alignment_CommonAlignment_AlignableObjectId_h
#define Alignment_CommonAlignment_AlignableObjectId_h

/// Type Identifier of Alignable objects (Det's, Rod's etc.).
/// Alignable has an AlignableObjectIdType as data member, which
/// is set in the constructor.
namespace AlignableObjectId {
  
  enum AlignableObjectIdType { 
    invalid                    =  0,
    AlignableDetUnit           =  1,
    AlignableDet               =  2,
    AlignableRod               =  3,
    AlignableBarrelLayer       =  4,
    AlignableHalfBarrel        =  5,
    AlignablePetal             =  6,
    AlignableEndcapLayer       =  7,
    AlignableEndcap            =  8,
    AlignableTIDRing           =  9,
    AlignableTIDLayer          = 10,
    AlignableTID               = 11,
    AlignablePxHalfBarrelLayer = 12,
    AlignablePxHalfBarrel      = 13,
    AlignableTracker           = 14,

    MuonAlignBM                = 20,
    MuonAlignHalfBM            = 21,
    MuonAlignBMLayer           = 22,
    MuonAlignSec               = 23,
    MuonAlignEnd               = 24,
    MuonAlignEndmLayer         = 25,
    MuonAlignAba               = 26
  };
}

#endif
