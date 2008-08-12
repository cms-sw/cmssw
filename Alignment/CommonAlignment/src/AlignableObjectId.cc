#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

using namespace align;

//__________________________________________________________________________________________________
AlignableObjectId::AlignableObjectId()
{

  // Names are defined here!

  theMap.clear();
  theReverseMap.clear();

  theMap.insert( PairEnumType( AlignableDetUnit, "DetUnit" ) );
  theMap.insert( PairEnumType( AlignableDet    , "Det"     ) );

  theMap.insert( PairEnumType(TPBModule      , "TPBModule"      ) );
  theMap.insert( PairEnumType(TPBLadder      , "TPBLadder"      ) );
  theMap.insert( PairEnumType(TPBLayer       , "TPBLayer"       ) );
  theMap.insert( PairEnumType(TPBHalfBarrel  , "TPBHalfBarrel"  ) );
  theMap.insert( PairEnumType(TPBBarrel      , "TPBBarrel"      ) );

  theMap.insert( PairEnumType(TPEModule      , "TPEModule"      ) );
  theMap.insert( PairEnumType(TPEPanel       , "TPEPanel"       ) );
  theMap.insert( PairEnumType(TPEBlade       , "TPEBlade"       ) );
  theMap.insert( PairEnumType(TPEHalfDisk    , "TPEHalfDisk"    ) );
  theMap.insert( PairEnumType(TPEHalfCylinder, "TPEHalfCylinder") );
  theMap.insert( PairEnumType(TPEEndcap      , "TPEEndcap"      ) );

  theMap.insert( PairEnumType(TIBModule      , "TIBModule"      ) );
  theMap.insert( PairEnumType(TIBString      , "TIBString"      ) );
  theMap.insert( PairEnumType(TIBSurface     , "TIBSurface"     ) );
  theMap.insert( PairEnumType(TIBHalfShell   , "TIBHalfShell"   ) );
  theMap.insert( PairEnumType(TIBLayer       , "TIBLayer"       ) );
  theMap.insert( PairEnumType(TIBHalfBarrel  , "TIBHalfBarrel"  ) );
  theMap.insert( PairEnumType(TIBBarrel      , "TIBBarrel"      ) );

  theMap.insert( PairEnumType(TIDModule      , "TIDModule"      ) );
  theMap.insert( PairEnumType(TIDSide        , "TIDSide"        ) );
  theMap.insert( PairEnumType(TIDRing        , "TIDRing"        ) );
  theMap.insert( PairEnumType(TIDDisk        , "TIDDisk"        ) );
  theMap.insert( PairEnumType(TIDEndcap      , "TIDEndcap"      ) );

  theMap.insert( PairEnumType(TOBModule      , "TOBModule"      ) );
  theMap.insert( PairEnumType(TOBRod         , "TOBRod"         ) );
  theMap.insert( PairEnumType(TOBLayer       , "TOBLayer"       ) );
  theMap.insert( PairEnumType(TOBHalfBarrel  , "TOBHalfBarrel"  ) );
  theMap.insert( PairEnumType(TOBBarrel      , "TOBBarrel"      ) );

  theMap.insert( PairEnumType(TECModule      , "TECModule"      ) );
  theMap.insert( PairEnumType(TECRing        , "TECRing"        ) );
  theMap.insert( PairEnumType(TECPetal       , "TECPetal"       ) );
  theMap.insert( PairEnumType(TECSide        , "TECSide"        ) );
  theMap.insert( PairEnumType(TECDisk        , "TECDisk"        ) );
  theMap.insert( PairEnumType(TECEndcap      , "TECEndcap"      ) );

  theMap.insert( PairEnumType(Pixel          , "Pixel"          ) );
  theMap.insert( PairEnumType(Strip          , "Strip"          ) );
  theMap.insert( PairEnumType(Tracker        , "Tracker"        ) );

  theMap.insert( PairEnumType( AlignableDTBarrel    ,  "DTBarrel"     ) );
  theMap.insert( PairEnumType( AlignableDTWheel     ,  "DTWheel"      ) );
  theMap.insert( PairEnumType( AlignableDTStation   ,  "DTStation"    ) );
  theMap.insert( PairEnumType( AlignableDTChamber   ,  "DTChamber"    ) );
  theMap.insert( PairEnumType( AlignableDTSuperLayer,  "DTSuperLayer" ) );
  theMap.insert( PairEnumType( AlignableDTLayer     ,  "DTLayer"      ) );
  theMap.insert( PairEnumType( AlignableCSCEndcap   ,  "CSCEndcap"    ) );
  theMap.insert( PairEnumType( AlignableCSCStation  ,  "CSCStation"   ) );
  theMap.insert( PairEnumType( AlignableCSCRing     ,  "CSCRing"      ) );
  theMap.insert( PairEnumType( AlignableCSCChamber  ,  "CSCChamber"   ) );
  theMap.insert( PairEnumType( AlignableCSCLayer    ,  "CSCLayer"     ) );
  theMap.insert( PairEnumType( AlignableMuon        ,  "Muon"         ) );

  // Create the reverse map
  std::transform( theMap.begin(), theMap.end(),
		  std::inserter( theReverseMap, theReverseMap.begin() ),
		  reverse_pair() );

}


//__________________________________________________________________________________________________
const StructureType
AlignableObjectId::nameToType( const std::string& name ) const
{
  ReverseMapEnumType::const_iterator n = theReverseMap.find(name);

  if (theReverseMap.end() == n)
  {
    throw cms::Exception("AlignableObjectIdError")
      << "Unknown alignableObjectId " << name;
  }

  return n->second;
}


//__________________________________________________________________________________________________
const std::string& AlignableObjectId::typeToName( StructureType type ) const
{
  MapEnumType::const_iterator t = theMap.find(type);

  if (theMap.end() == t)
  {
    throw cms::Exception("AlignableObjectIdError")
      << "Unknown alignableObjectId " << type;
  }

  return t->second;
}
