#ifndef GEOMETRY_CALOGEOMETRY_CALOSUBDETECTORGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOSUBDETECTORGEOMETRY_H 1

#include <vector>
#include <set>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"


/** \class CaloSubdetectorGeometry
      
Base class for a geometry container for a specific calorimetry
subdetector.


$Date: 2011/06/04 18:58:48 $
$Revision: 1.27 $
\author J. Mans - Minnesota
*/
class CaloSubdetectorGeometry {

   public:

      typedef CaloCellGeometry::CCGFloat CCGFloat ;

      typedef std::set<DetId>       DetIdSet;

      typedef CaloCellGeometry::ParMgr    ParMgr ;
      typedef CaloCellGeometry::ParVec    ParVec ;
      typedef CaloCellGeometry::ParVecVec ParVecVec ;

      typedef std::vector<CCGFloat>     TrVec  ;
      typedef std::vector<unsigned int> IVec   ;
      typedef std::vector<CCGFloat>     DimVec ;

      CaloSubdetectorGeometry() ;

      /// The base class DOES assume that it owns the CaloCellGeometry objects
      virtual ~CaloSubdetectorGeometry();

      virtual void newCell( const GlobalPoint& f1 ,
			    const GlobalPoint& f2 ,
			    const GlobalPoint& f3 ,
			    const CCGFloat*    parm ,
			    const DetId&       detId ) = 0 ;

      /// is this detid present in the geometry?
      virtual bool present( const DetId& id ) const;

      /// Get the cell geometry of a given detector id.  Should return false if not found.
      virtual const CaloCellGeometry* getGeometry( const DetId& id ) const ;

      /** \brief Get a list of valid detector ids (for the given subdetector)
	  \note The implementation in this class is relevant for SubdetectorGeometries which handle only
	  a single subdetector at a time.  It does not look at the det and subdet arguments.
      */
      virtual const std::vector<DetId>& getValidDetIds( DetId::Detector det    = DetId::Detector(0) , 
							int             subdet = 0                   ) const ;

      // Get closest cell, etc...
      virtual DetId getClosestCell( const GlobalPoint& r ) const ;

      /** \brief Get a list of all cells within a dR of the given cell
	  
      The default implementation makes a loop over all cell geometries.
      Cleverer implementations are suggested to use rough conversions between
      eta/phi and ieta/iphi and test on the boundaries.
      */
      virtual DetIdSet getCells( const GlobalPoint& r, double dR ) const ;

      CCGFloat deltaPhi( const DetId& detId ) const ;

      CCGFloat deltaEta( const DetId& detId ) const ;

      void allocateCorners( CaloCellGeometry::CornersVec::size_type n ) ;

      CaloCellGeometry::CornersMgr* cornersMgr() { return m_cmgr ; }

      void allocatePar( ParVec::size_type n, unsigned int m ) ;

      ParMgr* parMgr() { return m_parMgr ; }
      const ParMgr* parMgrConst() const { return m_parMgr ; }

      ParVecVec&       parVecVec()       { return m_parVecVec ; }
      const ParVecVec& parVecVec() const { return m_parVecVec ; }
      
      virtual unsigned int numberOfShapes() const { return 1 ; }
      virtual unsigned int numberOfParametersPerShape() const { return 1 ; }
      virtual unsigned int numberOfTransformParms() const { return 6 ; }

      virtual void fillDefaultNamedParameters() const { return ; }
      
      void getSummary( TrVec&  trVector,
		       IVec&   iVector,
		       DimVec& dimVector   ) const ;

      virtual void initializeParms() { return ; } 

   protected:

      virtual const CaloCellGeometry* cellGeomPtr( uint32_t index ) const = 0 ;

      ParVecVec m_parVecVec ;

      static CCGFloat deltaR( const GlobalPoint& p1,
			      const GlobalPoint& p2  ) 
      { return reco::deltaR( p1, p2 ) ; }

      mutable std::vector<DetId> m_validIds ;

   private:

      ParMgr*   m_parMgr ;

      CaloCellGeometry::CornersMgr* m_cmgr ;

      /// avoid copies
      CaloSubdetectorGeometry(            const CaloSubdetectorGeometry& ) ;
      CaloSubdetectorGeometry& operator=( const CaloSubdetectorGeometry& ) ;

      mutable bool m_sortedIds ;

      mutable std::vector<CCGFloat>*  m_deltaPhi ;
      mutable std::vector<CCGFloat>*  m_deltaEta ;
};


#endif
