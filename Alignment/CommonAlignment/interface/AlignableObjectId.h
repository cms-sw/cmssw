#ifndef Alignment_CommonAlignment_AlignableObjectId_h
#define Alignment_CommonAlignment_AlignableObjectId_h

/// Type Identifier of Alignable objects (Det's, Rod's etc.).
namespace AlignableObjectId 
{
  
  enum AlignableObjectIdType 
	{ 
	  invalid                    =  0,
	  AlignableDetUnit,
	  AlignableDet,
	  AlignableRod,
	  AlignableBarrelLayer,
	  AlignableHalfBarrel,
	  AlignablePetal,
	  AlignableEndcapLayer,
	  AlignableEndcap,
	  AlignableTIDRing,
	  AlignableTIDLayer,
	  AlignableTID,
	  AlignablePixelHalfBarrelLayer,
	  AlignablePixelHalfBarrel,
	  AlignableTracker,
	  
	  MuonAlignBM                = 20,
	  MuonAlignHalfBM,
	  MuonAlignBMLayer,
	  MuonAlignSec,
	  MuonAlignEnd,
	  MuonAlignEndmLayer,
	  MuonAlignAba

	};

}

#endif
