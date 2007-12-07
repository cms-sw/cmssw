/// \file TrackerAlignableId.cc
///
///  $Revision: 1.10 $
///  $Date: 2007/05/11 19:59:48 $
///  (last update by $Author: cklae $)

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"


#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"


//__________________________________________________________________________________________________
uint32_t TrackerAlignableId::alignableId( const Alignable* alignable ) const
{

  return alignable ? firstDetId(*alignable) : 0;

}



//__________________________________________________________________________________________________
// Get integer identifier corresponding to type of alignable
int TrackerAlignableId::alignableTypeId( const Alignable* alignable ) const
{

  int alignableObjectId = alignable->alignableObjectId();

  if ( AlignableObjectId::invalid == alignableObjectId ) 
	throw cms::Exception("LogicError") << "Unknown Alignable type";

  return alignableObjectId;

}

//_______________________________________________________________________
/// Return unique ID of alignable, consisting of the geographical ID of the
/// first GeomDet and the type ID (i.e. Rod, Layer, etc.) 
TrackerAlignableId::UniqueId TrackerAlignableId::alignableUniqueId( const Alignable* alignable ) const
{
  return UniqueId( alignableId(alignable), alignableTypeId(alignable) );
}


//__________________________________________________________________________________________________
// Returns alignable object id and layer number from an alignable
std::pair<int,int> TrackerAlignableId::typeAndLayerFromAlignable( const Alignable* alignable) const
{

  return alignable ? typeAndLayerFromDetId( firstDet(*alignable).geomDetId() ) : std::make_pair(0,0);

}


//__________________________________________________________________________________________________
// Returns alignable object id and layer (or wheel, or disk) number from a GeomDet
std::pair<int,int> TrackerAlignableId::typeAndLayerFromGeomDet( const GeomDet& geomDet ) const
{

  return typeAndLayerFromDetId( geomDet.geographicalId() );

}

//__________________________________________________________________________________________________
// Returns alignable object id and layer (or wheel, or disk) number from a DetId
std::pair<int,int> TrackerAlignableId::typeAndLayerFromDetId( const DetId& detId ) const
{

  int layerNumber = 0;

  unsigned int subdetId = static_cast<unsigned int>(detId.subdetId());

  if ( subdetId == StripSubdetector::TIB) 
	{ 
	  TIBDetId tibid(detId.rawId()); 
	  layerNumber = tibid.layer();
	}
  else if ( subdetId ==  StripSubdetector::TOB )
	{ 
	  TOBDetId tobid(detId.rawId()); 
	  layerNumber = tobid.layer();
	}
  else if ( subdetId ==  StripSubdetector::TID) 
	{ 
	  TIDDetId tidid(detId.rawId());
	  layerNumber = tidid.wheel();
	}
  else if ( subdetId ==  StripSubdetector::TEC )
	{ 
	  TECDetId tecid(detId.rawId()); 
	  layerNumber = tecid.wheel(); 
	}
  else if ( subdetId ==  PixelSubdetector::PixelBarrel ) 
	{ 
	  PXBDetId pxbid(detId.rawId()); 
	  layerNumber = pxbid.layer();  
	}
  else if ( subdetId ==  PixelSubdetector::PixelEndcap ) 
	{ 
	  PXFDetId pxfid(detId.rawId()); 
	  layerNumber = pxfid.disk();  
	}
  else
	edm::LogWarning("LogicError") << "Unknown subdetid: " <<  subdetId;


  return std::make_pair( subdetId, layerNumber );

}


//__________________________________________________________________________________________________
// Return string name corresponding to alignable
const std::string& TrackerAlignableId::alignableTypeName( const Alignable* alignable ) const
{
  if (alignable)
    return this->alignableTypeIdToName( alignable->alignableObjectId() );

  throw cms::Exception("LogicError") << "Alignable=0";

}

//__________________________________________________________________________________________________
const std::string&
TrackerAlignableId::alignableTypeIdToName( int id ) const
{

  static AlignableObjectId alignableObjectId;
  return alignableObjectId.typeToName( id );

}

//__________________________________________________________________________________________________
// recursively get first Alignable Det of an Alignable
const Alignable& TrackerAlignableId::firstDet( const Alignable& alignable ) const
{

  if ( AlignableObjectId::AlignableDet == alignable.alignableObjectId() ||
       AlignableObjectId::AlignableDetUnit == alignable.alignableObjectId() )
    return alignable;

  return firstDet( *alignable.components().front() );

}

//__________________________________________________________________________________________________
// get integer identifier corresponding to 1st Det of alignable
uint32_t TrackerAlignableId::firstDetId( const Alignable& alignable ) const
{

  return firstDet( alignable ).geomDetId().rawId();

}


