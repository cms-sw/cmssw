#ifndef Alignment_CommonAlignment_AlignableObjectId_h
#define Alignment_CommonAlignment_AlignableObjectId_h

#include <map>
#include <string>

/// Type Identifier of Alignable objects (Det's, Rod's etc.).
/// Allows conversion between type and name, and vice-versa
class AlignableObjectId 
{

public:
  
  /// Constructor (defines names)
  AlignableObjectId();

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
  
  typedef std::map<AlignableObjectIdType, std::string> MapEnumType;
  typedef std::map<std::string, AlignableObjectIdType> ReverseMapEnumType;
  typedef std::pair<AlignableObjectIdType, std::string> PairEnumType;
  typedef std::pair<std::string, AlignableObjectIdType> PairEnumReverseType;

  /// Convert name to type
  const AlignableObjectIdType nameToType( const std::string name ) const;

  /// Convert type to name
  const std::string typeToName( const int type ) const;

private:
  MapEnumType theMap;
  ReverseMapEnumType theReverseMap;

  // Reverse functor
  struct reverse_pair {
	PairEnumReverseType operator()( const PairEnumType& pair ) const 
	{ 
	  return PairEnumReverseType( pair.second, pair.first ); 
	}
  };


};

#endif
