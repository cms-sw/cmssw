#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//__________________________________________________________________________________________________
AlignableObjectId::AlignableObjectId( void )
{

  // Names are defined here!

  theMap.clear();
  theReverseMap.clear();

  theMap.insert( PairEnumType( AlignableDetUnit              ,  "DetUnit"              ) );
  theMap.insert( PairEnumType( AlignableDet                  ,  "Det"                  ) );
  theMap.insert( PairEnumType( AlignableRod                  ,  "Rod"                  ) );
  theMap.insert( PairEnumType( AlignableBarrelLayer          ,  "BarrelLayer"          ) );
  theMap.insert( PairEnumType( AlignableHalfBarrel           ,  "HalfBarrel"           ) );
  theMap.insert( PairEnumType( AlignablePetal                ,  "Petal"                ) );
  theMap.insert( PairEnumType( AlignableEndcapLayer          ,  "EndcapLayer"          ) );
  theMap.insert( PairEnumType( AlignableEndcap               ,  "Endcap"               ) );
  theMap.insert( PairEnumType( AlignableTIDRing              ,  "TIDRing"              ) );
  theMap.insert( PairEnumType( AlignableTIDLayer             ,  "TIDLayer"             ) );
  theMap.insert( PairEnumType( AlignableTID                  ,  "TID"                  ) );
  theMap.insert( PairEnumType( AlignablePixelHalfBarrelLayer ,  "PixelHalfBarrelLayer" ) );
  theMap.insert( PairEnumType( AlignablePixelHalfBarrel      ,  "PixelHalfBarrel"      ) );
  theMap.insert( PairEnumType( AlignableTracker              ,  "Tracker"              ) );

  theMap.insert( PairEnumType( AlignableDTBarrel             ,  "DTBarrel"             ) );
  theMap.insert( PairEnumType( AlignableDTWheel              ,  "DTWheel"              ) );
  theMap.insert( PairEnumType( AlignableDTStation            ,  "DTStation"            ) );
  theMap.insert( PairEnumType( AlignableDTChamber            ,  "DTChamber"            ) );
  theMap.insert( PairEnumType( AlignableDTSuperLayer         ,  "DTSuperLayer"         ) );
  theMap.insert( PairEnumType( AlignableDTLayer              ,  "DTLayer"              ) );
  theMap.insert( PairEnumType( AlignableCSCEndcap            ,  "CSCEndcap"            ) );
  theMap.insert( PairEnumType( AlignableCSCStation           ,  "CSCStation"           ) );
  theMap.insert( PairEnumType( AlignableCSCChamber           ,  "CSCChamber"           ) );
  theMap.insert( PairEnumType( AlignableCSCLayer             ,  "CSCLayer"             ) );
  theMap.insert( PairEnumType( AlignableMuon                 ,  "Muon"                 ) );

  // Create the reverse map
  std::transform( theMap.begin(), theMap.end(),
				  std::inserter( theReverseMap, theReverseMap.begin() ),
				  reverse_pair() );

}


//__________________________________________________________________________________________________
const AlignableObjectId::AlignableObjectIdType
AlignableObjectId::nameToType( const std::string name ) const
{
  if ( theReverseMap.find(name) != theReverseMap.end() ) 
 	return ( theReverseMap.find(name) )->second;

  return invalid;
   
}


//__________________________________________________________________________________________________
const std::string AlignableObjectId::typeToName( const int type ) const
{
  AlignableObjectIdType m_IdType = AlignableObjectIdType( type );

  if ( theMap.find(m_IdType) != theMap.end() ) return ( theMap.find(m_IdType) )->second;

  edm::LogError("LogicError") << "Unknown alignableObjectId " << type;
  return "INVALID";

}
