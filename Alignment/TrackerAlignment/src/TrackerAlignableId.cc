/// \file TrackerAlignableId.cc
///
///  $Revision: 1.13 $
///  $Date: 2013/01/07 19:44:30 $
///  (last update by $Author: wmtan $)

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"


//__________________________________________________________________________________________________
// Returns alignable object id and layer (or wheel, or disk) number from a DetId
std::pair<int,int> TrackerAlignableId::typeAndLayerFromDetId( const DetId& detId , const TrackerTopology* tTopo) const
{

  int layerNumber = 0;

  unsigned int subdetId = static_cast<unsigned int>(detId.subdetId());

  if ( subdetId == StripSubdetector::TIB) 
	{ 
	   
	  layerNumber = tTopo->tibLayer(detId.rawId());
	}
  else if ( subdetId ==  StripSubdetector::TOB )
	{ 
	   
	  layerNumber = tTopo->tobLayer(detId.rawId());
	}
  else if ( subdetId ==  StripSubdetector::TID) 
	{ 
	  
	  layerNumber = tTopo->tidWheel(detId.rawId());
	}
  else if ( subdetId ==  StripSubdetector::TEC )
	{ 
	   
	  layerNumber = tTopo->tecWheel(detId.rawId()); 
	}
  else if ( subdetId ==  PixelSubdetector::PixelBarrel ) 
	{ 
	   
	  layerNumber = tTopo->pxbLayer(detId.rawId());  
	}
  else if ( subdetId ==  PixelSubdetector::PixelEndcap ) 
	{ 
	   
	  layerNumber = tTopo->pxfDisk(detId.rawId());  
	}
  else
	edm::LogWarning("LogicError") << "Unknown subdetid: " <<  subdetId;


  return std::make_pair( subdetId, layerNumber );

}
