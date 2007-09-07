#ifndef GEOMETRY_CALOGEOMETRY_CALOSUBDETECTORGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOSUBDETECTORGEOMETRY_H 1

#include <ext/hash_map>
#include <vector>
#include <set>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

/** \class CaloSubdetectorGeometry
      
Base class for a geometry container for a specific calorimetry
subdetector.


$Date: 2007/09/05 19:53:08 $
$Revision: 1.11 $
\author J. Mans - Minnesota
*/
class CaloSubdetectorGeometry {

   public:

      typedef  __gnu_cxx::hash_map< unsigned int, const CaloCellGeometry *> CellCont;

      typedef std::set<DetId>       DetIdSet;

      typedef std::vector<float>  ParVec ;
      typedef std::vector<ParVec> ParVecVec ;

      CaloSubdetectorGeometry() : m_cmgr   ( 0 ) {}

      /// The base class DOES assume that it owns the CaloCellGeometry objects
      virtual ~CaloSubdetectorGeometry();

   public:
      /// the cells
      const CellCont& cellGeometries() const { return m_cellG ; }  

      /// Add a cell to the geometry
      void addCell( const DetId& id, 
		    const CaloCellGeometry* ccg ) ;

      /// is this detid present in the geometry?
      virtual bool present( const DetId& id ) const;

      /// Get the cell geometry of a given detector id.  Should return false if not found.
      virtual const CaloCellGeometry* getGeometry( const DetId& id ) const ;

      /** \brief Get a list of valid detector ids (for the given subdetector)
	  \note The implementation in this class is relevant for SubdetectorGeometries which handle only
	  a single subdetector at a time.  It does not look at the det and subdet arguments.
      */
      virtual const std::vector<DetId>& getValidDetIds( DetId::Detector det, 
							int subdet  ) const ;

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

      ParVecVec& parVecVec() { return m_parVecVec ; }

   protected:

      ParVecVec m_parVecVec ;

      mutable std::vector<DetId> m_validIds ;

      static double deltaR( const GlobalPoint& p1,
			    const GlobalPoint& p2  ) ;

   private:

      CaloCellGeometry::CornersMgr* m_cmgr ;

      /// avoid copies
      CaloSubdetectorGeometry(            const CaloSubdetectorGeometry& ) ;
      CaloSubdetectorGeometry& operator=( const CaloSubdetectorGeometry& ) ;

      CellCont m_cellG ;    

};


#endif
