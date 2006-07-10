#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"


#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"


//__________________________________________________________________________________________________
TrackerAlignableId::TrackerAlignableId()
{
}


//__________________________________________________________________________________________________
unsigned int TrackerAlignableId::alignableId( Alignable* alignable )
{

  return firstDetId(alignable);

}



//__________________________________________________________________________________________________
// Get integer identifier corresponding to type of alignable
int TrackerAlignableId::alignableTypeId( Alignable* alignable )
{

  int alignableObjectId = static_cast<int>(alignable->alignableObjectId());

  if ( !alignableObjectId ) 
	throw cms::Exception("LogicError") << "Unknown Alignable type";

  return alignableObjectId;

}





//__________________________________________________________________________________________________
// Returns alignable object id and layer number from an alignable
std::pair<int,int> TrackerAlignableId::typeAndLayerFromAlignable(Alignable* alignable)
{

  if ( alignable ) 
	{
	  AlignableDet* alignableDet = firstDet(alignable);
	  GeomDet* geomDet = alignableDet->geomDet();
	  if ( alignableDet ) 
		return typeAndLayerFromGeomDet( *geomDet );
	}

  return std::make_pair(0,0);

}


//__________________________________________________________________________________________________
// Returns alignable object id and layer (or wheel, or disk) number from a GeomDet
std::pair<int,int> TrackerAlignableId::typeAndLayerFromGeomDet( const GeomDet& geomDet )
{

  int layerNumber = 0;

  DetId id = geomDet.geographicalId();
  unsigned int subdetId = static_cast<unsigned int>(id.subdetId());

  if ( subdetId == StripSubdetector::TIB) 
	{ 
	  TIBDetId tibid(id.rawId()); 
	  layerNumber = tibid.layer();
	}
  else if ( subdetId ==  StripSubdetector::TOB )
	{ 
	  TOBDetId tobid(id.rawId()); 
	  layerNumber = tobid.layer();
	}
  else if ( subdetId ==  StripSubdetector::TID) 
	{ 
	  TIDDetId tidid(id.rawId());
	  layerNumber = tidid.wheel();
	}
  else if ( subdetId ==  StripSubdetector::TEC )
	{ 
	  TECDetId tecid(id.rawId()); 
	  layerNumber = tecid.wheel(); 
	}
  else if ( subdetId ==  PixelSubdetector::PixelBarrel ) 
	{ 
	  PXBDetId pxbid(id.rawId()); 
	  layerNumber = pxbid.layer();  
	}
  else if ( subdetId ==  PixelSubdetector::PixelEndcap ) 
	{ 
	  PXFDetId pxfid(id.rawId()); 
	  layerNumber = pxfid.disk();  
	}
  else
	edm::LogWarning("LogicError") << "Unknown subdetid: " <<  subdetId;


  return std::make_pair( subdetId, layerNumber );

}


//__________________________________________________________________________________________________
// Return string name corresponding to alignable
const std::string TrackerAlignableId::alignableTypeName( const Alignable* alignable ) const
{

  return this->alignableTypeIdToName( alignable->alignableObjectId() );

}

//__________________________________________________________________________________________________
const std::string 
TrackerAlignableId::alignableTypeIdToName( const int& id ) const
{

  AlignableObjectId alignableObjectId;
  return alignableObjectId.typeToName( id );

}

//__________________________________________________________________________________________________
// recursively get first Alignable Det of an Alignable
AlignableDet* TrackerAlignableId::firstDet( Alignable* alignable )
{

  // Check if this is already an AlignableDet
  AlignableDet* alignableDet = dynamic_cast<AlignableDet*>( alignable );
  if ( alignableDet ) return ( alignableDet );

  // Otherwise, retrieve components
  AlignableComposite* composite = dynamic_cast<AlignableComposite*>( alignable );
  return  firstDet( composite->components().front() );

}



//__________________________________________________________________________________________________
// get integer identifier corresponding to 1st Det of alignable
unsigned int TrackerAlignableId::firstDetId( Alignable* alignable )
{

  unsigned int geomDetId = 0;

  if ( alignable ) 
	{
	  AlignableDet* alignableDet = firstDet( alignable );
	  if ( alignableDet ) geomDetId = alignableDet->geomDet()->geographicalId().rawId();
	}

  return geomDetId;

}


