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
	  AlignableHalfBarrel,       // 5
	  AlignablePetal,
	  AlignableEndcapLayer,
	  AlignableEndcap,
	  AlignableTIDRing,
	  AlignableTIDLayer,         // 10
	  AlignableTID,
	  AlignablePixelHalfBarrelLayer,
	  AlignablePixelHalfBarrel,
	  AlignableTracker,
	  
	  AlignableDTBarrel              = 20,
	  AlignableDTWheel,
	  AlignableDTStation,
	  AlignableDTChamber,
	  AlignableDTSuperLayer,
	  AlignableDTLayer,          // 25
	  AlignableCSCEndcap,
	  AlignableCSCStation,
	  AlignableCSCChamber,
	  AlignableCSCLayer,
	  AlignableMuon
	  
	};

}

#endif
