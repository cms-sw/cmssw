/// \file TrackerAlignableId.cc
///
///  $Revision: 1.12 $
///  $Date: 2007/10/08 13:49:07 $
///  (last update by $Author: cklae $)

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"


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
