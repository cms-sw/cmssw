#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerDetIdBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <bitset>

CmsTrackerDetIdBuilder::CmsTrackerDetIdBuilder( unsigned int layerNumberPXB )
  : m_layerNumberPXB( layerNumberPXB )
{}

GeometricDet*
CmsTrackerDetIdBuilder::buildId( GeometricDet* tracker )
{
  DetId t( DetId::Tracker, 0 );
  tracker->setGeographicalID( t );
  iterate( tracker, 0, tracker->geographicalID().rawId() );

  return tracker;
}

void
CmsTrackerDetIdBuilder::iterate( GeometricDet const *in, int level, unsigned int ID )
{
  std::bitset<32> binary_ID(ID);

  // SubDetector (useful to know fron now on, valid only after level 0, where SubDetector is assigned)
  uint32_t mask = (7<<25);
  uint32_t iSubDet = ID & mask;
  iSubDet = iSubDet >> 25;
  //
  
  switch( level )
  {
    // level 0
  case 0:
    {  
      for( uint32_t i = 0; i<(in)->components().size(); i++ )
      {
	uint32_t jSubDet = ((in)->components())[i]->geographicalID().rawId();
	uint32_t temp = ID;
	temp |= (jSubDet<<25);
	((in)->components())[i]->setGeographicalID(temp);	
	
	switch( jSubDet )
	{ 
	  // PXF
	case 2:
	  {
	    // SubDetector Side start bit is 23 [3 unused and Side length is 2 bit]
	    if(((in)->components())[i]->translation().z()<0. )
	    {
	      temp |= (1<<23); // PXF-
	    }
	    else
	    {
	      temp |= (2<<23); // PXF+
	    }
	    break;
	  }
	  // TID
	case 4:
	  {
	    temp|= (0<<15); // SubDetector Side start bit is 13 [10 unused and Side length is 2 bit]
	    if((((in)->components())[i])->components()[0]->translation().z()<0. )
	    {
	      temp |= (1<<13); // TIDB = TID-
	    }
	    else
	    {
	      temp |= (2<<13); // TIDF = TID+
	    }
	    break;
	  }
	  // TEC
	case 6:
	  {
	    temp |= (0<<20); // SubDetector Side start bit is 18 [5 unused and Side length is 2 bit]
	    if(((in)->components())[i]->translation().z()<0. )
	    {
	      temp |= (1<<18); // TEC-
	    }
	    else
	    {
	      temp |= (2<<18); // TEC+
	    }
	    break;
	  }
	  // PXB, TIB, TOB (barrel)
	default:
	  {
	    // do nothing
	  }
	  // SubDetector switch ends
	}
	
	((in)->components())[i]->setGeographicalID(DetId(temp));	
	
	// next level
	iterate(((in)->components())[i],level+1,((in)->components())[i]->geographicalID().rawId());
      }
      break;
    }
  // level 1
  case 1:
    {
      for( uint32_t i = 0; i < (in)->components().size(); i++ )
      {
	uint32_t temp = ID;
      
	switch( iSubDet )
	{
	// PXB
	case 1:
	  {
//	    temp |= (((in)->components())[i]->geographicalID().rawId() << m_layerNumberPXB ); // Layer Number start bit is 16 [5 unused]
	    temp |= (((in)->components())[i]->geographicalID().rawId()<<20); // Layer Number start bit is 16 [5 unused]
	    break;
	  }
	// PXF
	case 2:
	  {
//	    temp |= (((in)->components())[i]->geographicalID().rawId() << 16 ); // Disk Number start bit is 16
            temp |= (((in)->components())[i]->geographicalID().rawId()<<18); // Disk Number start bit is 16
	    break;
	  }
	// TIB
	case 3:
	  {
	    temp |= (((in)->components())[i]->geographicalID().rawId() << 14); // Layer Number start bit is 14 [8 unused]
	    break;
	  }
	
	// TID
	case 4:
	  {
	    temp |= (((in)->components())[i]->geographicalID().rawId() << 11); // Disk (Wheel) Number start bit is 11
	    break;
	  }
	
	// TOB
	case 5:
	  {
	    temp |= (((in)->components())[i]->geographicalID().rawId() << 14); // Layer Number start bit is 14 [8 unused]
	    break;
	  }
	
	// TEC
	case 6:
	  {
	    temp |= (((in)->components())[i]->geographicalID().rawId() << 14); // Wheel Number start bit is 14
	    break;
	  }
      
	// the rest
	default:
	  {
	    // do nothing
	  }
	
	// SubDetector switch ends
	}
      
	((in)->components())[i]->setGeographicalID( temp );
      
	// next level
	iterate(((in)->components())[i],level+1,((in)->components())[i]->geographicalID().rawId());      
      }
    
      break; 
    }    
    // level 2
  case 2: {
    
    for (uint32_t i=0;i<(in)->components().size();i++) {
	
      switch (iSubDet) {

	// PXB
      case 1:
	{
	  uint32_t temp = ID;
	  // Ladder Starting bit = 2 (last unused) + 6 (Module Number) = 8
	  temp |= (((in)->components())[i]->geographicalID().rawId()<<12);//<<10);//8)
	  ((in)->components())[i]->setGeographicalID(temp);
	  break;
	}
	
	// PXF
      case 2:
	{
	  uint32_t temp = ID;
	  // Blade Starting bit = 1 (last unused) + 5 (Module Number) + 2 (Plaquette part) = 8
//	  temp |= (((in)->components())[i]->geographicalID().rawId()<<8);
          temp |= (((in)->components())[i]->geographicalID().rawId()<<10);
	  ((in)->components())[i]->setGeographicalID(temp);
	  break;
	}
	
	// TIB
      case 3:
	{
	  uint32_t temp = ID;
	  // Side+Part+String Starting bit = 2 (Module Type) + 2 (Module Number) = 4
	  temp |= (((in)->components())[i]->geographicalID().rawId()<<4);
	  ((in)->components())[i]->setGeographicalID(temp);
	  break;
	}
	
	// TID
      case 4:
	{
	  uint32_t temp = ID;
	  // Ring+Part Starting bit = 2 (Module Type) + 5 (Module Number) + 2 (Disk Part)= 9
	  temp |= (((in)->components())[i]->geographicalID().rawId()<<9);
	  ((in)->components())[i]->setGeographicalID(DetId(temp));
	  break;
	}
	
	// TOB
      case 5:
	{
	  uint32_t temp = ID;
	  // Side+Rod Starting bit = 2 (Module Type) + 3 (Module Number) = 5
	  temp |= (((in)->components())[i]->geographicalID().rawId()<<5);
	  ((in)->components())[i]->setGeographicalID(temp);
	  break;
	}
	
	// TEC
      case 6:
	{
	  uint32_t temp = ID;
	  // Petal+Part Starting bit = 2 (Module Type) + 3 (Module Number) + 3 (Ring Number) = 8
	  temp |= (((in)->components())[i]->geographicalID().rawId()<<8);
	  ((in)->components())[i]->setGeographicalID(temp);
	  break;
	}
	
	// the rest
      default:
	{
	  // do nothing
	}
	
	// SubDetector switch ends
      }

      // next level
      iterate(((in)->components())[i],level+1,((in)->components())[i]->geographicalID().rawId());
    }
    
    break;    
  }
    
    // level 3
  case 3:
    {
      for (uint32_t i=0;i<(in)->components().size();i++) {
	
	switch (iSubDet) {
	  
	  // TEC
	case 6:
	  {
	    // Ring Starting bit = 2 (Module Type) + 3 (Module Number)
	    uint32_t temp = ID;
	    temp |= (((in)->components())[i]->geographicalID().rawId()<<5);
	    ((in)->components())[i]->setGeographicalID(temp);
	    break;
	  }
	  
	  // the others but TEC
	default:
	  {
	    uint32_t temp = ID;
	    temp |= (((in)->components())[i]->geographicalID().rawId()<<2); // Starting bit = 2 (Module Type)
	    ((in)->components())[i]->setGeographicalID(temp);
	  }
	  
	  // SubDetector switch ends
	}
	
	// next level
	iterate(((in)->components())[i],level+1,((in)->components())[i]->geographicalID().rawId());
	
      }
      
      break;
    }
    
    // level 4
  case 4:
    {

      for (uint32_t i=0;i<(in)->components().size();i++) {
	
	switch (iSubDet) {
	  
	  // TEC
	case 6:
	  {
	    // Module Number bit = 2 (Module Type)
	    uint32_t temp = ID;
	    temp |= (((in)->components())[i]->geographicalID().rawId()<<2);
	    ((in)->components())[i]->setGeographicalID(temp);
	    // next level
	    iterate(((in)->components())[i],level+1,((in)->components())[i]->geographicalID().rawId());
	    break;
	  }
	  
	  // the others stop here!
	default:
	  {
	    for (uint32_t i=0;i<(in)->components().size();i++) {
	      uint32_t temp = ID;
	      temp |= (((in)->components())[i]->geographicalID().rawId());
	      ((in)->components())[i]->setGeographicalID(temp);
	    }
	  }
	  
	  // SubDetector switch ends
	}

      }
      
      break;
    }
    
    // level 5
  case 5:
    {
      // TEC Module Type (only TEC arrives here)
      for (uint32_t i=0;i<(in)->components().size();i++) {
	uint32_t temp = ID;
	temp |= (((in)->components())[i]->geographicalID().rawId());
	((in)->components())[i]->setGeographicalID(temp);
      }
      break;
    }
    
    // throw exception
  default:
    {
      cms::Exception("LogicError") <<" CmsTrackerDetIdBuilder invalid level "<< level;
    }
    
    // level switch ends
  }
  
  return;

}

