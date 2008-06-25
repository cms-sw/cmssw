#ifndef GEOMETRY_CALOGEOMETRY_CALOSUBDETECTORGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOSUBDETECTORGEOMETRY_H 1

#include <ext/hash_map>
#include <vector>
#include <set>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "PhysicsTools/Utilities/interface/deltaR.h"

/** \class CaloSubdetectorGeometry
      
Base class for a geometry container for a specific calorimetry
subdetector.


$Date: 2008/02/29 14:52:51 $
$Revision: 1.15 $
\author J. Mans - Minnesota
*/
class CaloSubdetectorGeometry {

   public:

      typedef  std::vector< const CaloCellGeometry * > CellCont;

      typedef std::set<DetId>       DetIdSet;

      typedef CaloCellGeometry::ParMgr    ParMgr ;
      typedef CaloCellGeometry::ParVec    ParVec ;
      typedef CaloCellGeometry::ParVecVec ParVecVec ;

      CaloSubdetectorGeometry() : 
	 m_parMgr ( 0 ) ,
	 m_cmgr   ( 0 )   {}

      /// The base class DOES assume that it owns the CaloCellGeometry objects
      virtual ~CaloSubdetectorGeometry();

      /// the cells
      const CellCont& cellGeometries() const { return m_cellG ; }  

      /// Add a cell to the geometry
      void addCell( const DetId&            id, 
		    const CaloCellGeometry* ccg ) ;

      /// is this detid present in the geometry?
      virtual bool present( const DetId& id ) const;

      /// Get the cell geometry of a given detector id.  Should return false if not found.
      virtual const CaloCellGeometry* getGeometry( const DetId& id ) const ;

      /** \brief Get a list of valid detector ids (for the given subdetector)
	  \note The implementation in this class is relevant for SubdetectorGeometries which handle only
	  a single subdetector at a time.  It does not look at the det and subdet arguments.
      */
      virtual const std::vector<DetId>& getValidDetIds( DetId::Detector det    = DetId::Detector(0) , 
							int             subdet = 0  ) const 
      { return m_validIds ; }

      // Get closest cell, etc...
      virtual DetId getClosestCell( const GlobalPoint& r ) const ;

      /** \brief Get a list of all cells within a dR of the given cell
	  
      The default implementation makes a loop over all cell geometries.
      Cleverer implementations are suggested to use rough conversions between
      eta/phi and ieta/iphi and test on the boundaries.
      */
      virtual DetIdSet getCells( const GlobalPoint& r, double dR ) const ;

      //FIXME: Hcal implements its own  getValidDetId....

      void allocateCorners( CaloCellGeometry::CornersVec::size_type n ) ;

      CaloCellGeometry::CornersMgr* cornersMgr() { return m_cmgr ; }

      void allocatePar( ParVec::size_type n, unsigned int m ) ;

      ParMgr* parMgr() { return m_parMgr ; }

      ParVecVec& parVecVec() { return m_parVecVec ; }

   protected:

      ParVecVec m_parVecVec ;

      static double deltaR( const GlobalPoint& p1,
			    const GlobalPoint& p2  ) 
      { return reco::deltaR( p1, p2 ) ; }

   private:

      ParMgr*   m_parMgr ;

      CaloCellGeometry::CornersMgr* m_cmgr ;

      /// avoid copies
      CaloSubdetectorGeometry(            const CaloSubdetectorGeometry& ) ;
      CaloSubdetectorGeometry& operator=( const CaloSubdetectorGeometry& ) ;

      CellCont m_cellG ;    

      std::vector<DetId> m_validIds ;
};


#endif
