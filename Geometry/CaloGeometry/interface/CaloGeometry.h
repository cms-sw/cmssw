#ifndef GEOMETRY_CALOGEOMETRY_CALOGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOGEOMETRY_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class CaloSubdetectorGeometry;
class CaloCellGeometry;

/** \class CaloGeometry
      
Interface class to the geometry information for all calorimetry 
subdetectors.

$Revision: 1.7 $
\author J. Mans and P. Meridiani
*/

class CaloGeometry 
{
   public:

      CaloGeometry() ;
    
      /// Register a subdetector geometry
      void setSubdetGeometry( DetId::Detector                det    , 
			      int                            subdet , 
			      const CaloSubdetectorGeometry* geom    ) ;
    
      /// Get the position of a given detector id
      const GlobalPoint& getPosition( const DetId& id ) const ;

      /// Get the cell geometry of a given detector id
      const CaloCellGeometry* getGeometry( const DetId& id ) const ;

      /// Get the list of all valid detector ids
      std::vector<DetId> getValidDetIds() const ;
    
      /// Get the list of valid detector ids for the given subdetector
      const std::vector<DetId>& getValidDetIds( DetId::Detector det, int subdet ) const;

      /// is this detid present in the geometry?
      bool present( const DetId& id ) const ;

      /// access the subdetector geometry for the given subdetector directly
      const CaloSubdetectorGeometry* getSubdetectorGeometry( const DetId& id ) const ;
    
      /// access the subdetector geometry for the given subdetector directly
      const CaloSubdetectorGeometry* getSubdetectorGeometry( DetId::Detector det    , 
							     int             subdet   ) const ;

      // Get closest cell, etc...
//not implemented      const DetId getClosestCell( const GlobalPoint& r ) const ;

   private:

      static const std::vector<DetId> k_emptyVec ;

      std::vector< const CaloSubdetectorGeometry* > m_geos ;

      unsigned int makeIndex( DetId::Detector det,
			      int             subdet,
			      bool&           ok      ) const ;

      enum { kMaxDet = 5 ,
	     kMinDet = 3 ,
	     kNDets  = kMaxDet - kMinDet + 1 ,
	     kMaxSub = 6 ,
	     kLength = kNDets*kMaxSub } ;
};



#endif
