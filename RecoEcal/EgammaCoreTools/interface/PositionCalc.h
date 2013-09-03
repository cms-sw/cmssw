#ifndef RecoEcal_EgammaCoreTools_PositionCalc_h
#define RecoEcal_EgammaCoreTools_PositionCalc_h

/** \class PositionClac
 *  
 * Finds the position and covariances for a cluster 
 * Formerly LogPositionCalc
 *
 * \author Ted Kolberg, ND
 * 
 *
 */

#include <vector>
#include <map>

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

class PositionCalc
{
 public:
  // You must call Initialize before you can calculate positions or 
  // covariances.

  PositionCalc(const edm::ParameterSet& par);
  PositionCalc() { };

  const PositionCalc& operator=(const PositionCalc& rhs);

  // Calculate_Location calculates an arithmetically or logarithmically
  // weighted average position of a vector of DetIds, which should be
  // a subset of the map used to Initialize.

  template<typename HitType>
  math::XYZPoint Calculate_Location( const std::vector< std::pair< DetId, float > >&      iDetIds  ,
				     const edm::SortedCollection<HitType>*    iRecHits ,
				     const CaloSubdetectorGeometry* iSubGeom ,
				     const CaloSubdetectorGeometry* iESGeom = 0 ) ;

 private:
  bool    param_LogWeighted_;
  double  param_T0_barl_;
  double  param_T0_endc_;
  double  param_T0_endcPresh_;
  double  param_W0_;
  double  param_X0_;

  const CaloSubdetectorGeometry* m_esGeom ;
  bool m_esPlus ;
  bool m_esMinus ;

};

template<typename HitType>
math::XYZPoint 
PositionCalc::Calculate_Location( const std::vector< std::pair<DetId, float> >&      iDetIds  ,
				  const edm::SortedCollection<HitType>*    iRecHits ,
				  const CaloSubdetectorGeometry* iSubGeom ,
				  const CaloSubdetectorGeometry* iESGeom   )
{
  typedef edm::SortedCollection<HitType> HitTypeCollection;
  math::XYZPoint returnValue ( 0, 0, 0 ) ;
  
  // Throw an error if the cluster was not initialized properly
  
  if( 0 == iRecHits || 
      0 == iSubGeom    )
    {
      throw(std::runtime_error("\n\nPositionCalc::Calculate_Location called uninitialized or wrong initialization.\n\n"));
    }
  
  if( 0 != iDetIds.size()   &&
      0 != iRecHits->size()     )
   {
     typedef std::vector< std::pair<DetId,float> > DetIdVec ; 
     DetIdVec detIds ;
     detIds.reserve( iDetIds.size() ) ;
     
     double eTot  ( 0 ) ;
     double eMax  ( 0 ) ;
     DetId  maxId ;
     
     // Check that DetIds are nonzero
     typename HitTypeCollection::const_iterator endRecHits ( iRecHits->end() ) ;
     for( std::vector< std::pair<DetId, float> >::const_iterator n ( iDetIds.begin() ) ; n != iDetIds.end() ; ++n ) 
       {
	 const DetId dId ( (*n).first ) ;
         const float frac( (*n).second) ; 
	 if( !dId.null() )
	   {
	     typename HitTypeCollection::const_iterator iHit ( iRecHits->find( dId ) ) ;
	     if( iHit != endRecHits )
	       {	       
		 detIds.push_back( std::pair<DetId,float>(dId,frac) );  
		 const double energy ( iHit->energy() *frac ) ;
		 		 
		 if( 0 < energy ) // only save positive energies
		   {
		     if( eMax < energy )
		       {
			 eMax  = energy ;
			 maxId = dId    ;
		       }
		     eTot += energy ;
		   }
	       }
	   }
       }
     
     if( 0 >= eTot )
       {
	 LogDebug("ZeroClusterEnergy") << "cluster with 0 energy: " 
				       << eTot << " size: " << detIds.size() 
				       << " , returning (0,0,0)";
       }
     else
       {
	 // first time or when es geom changes set flags
	 if( 0        != iESGeom &&
	     m_esGeom != iESGeom    )
	   {
	     m_esGeom = iESGeom ;
	     for( uint32_t ic ( 0 ) ;
		  ( ic != m_esGeom->getValidDetIds().size() ) &&
		    ( (!m_esPlus) || (!m_esMinus) ) ; ++ic )
	       {
		 const double z ( m_esGeom->getGeometry( m_esGeom->getValidDetIds()[ic] )->getPosition().z() ) ;
		 m_esPlus  = m_esPlus  || ( 0 < z ) ;
		 m_esMinus = m_esMinus || ( 0 > z ) ;
	       }
	   }
	 
	 //Select the correct value of the T0 parameter depending on subdetector
	 
	 const CaloCellGeometry* center_cell ( iSubGeom->getGeometry( maxId ) ) ;
	 const double ctreta (center_cell->getPosition().eta());
	 
	 // for barrel, use barrel T0; 
	 // for endcap: if preshower present && in preshower fiducial, 
         //             use preshower T0
	 // else use endcap only T0
	 
	 const double preshowerStartEta =  1.653;
	 const int subdet = maxId.subdetId();
	 const double T0 ( subdet == EcalBarrel ? param_T0_barl_ :
			   ( ( ( preshowerStartEta < fabs( ctreta ) ) &&
			       ( ( ( 0 < ctreta ) && 
				   m_esPlus          ) ||
				 ( ( 0 > ctreta ) &&
				   m_esMinus         )   )    ) ?
			     param_T0_endcPresh_ : param_T0_endc_ ) ) ;
	 
	 // Calculate shower depth
	 const float maxDepth ( param_X0_ * ( T0 + log( eTot ) ) ) ;
	 const float maxToFront ( center_cell->getPosition().mag() ) ; // to front face
	 
	 // Loop over hits and get weights
	 double total_weight = 0;
	 
	 double xw ( 0 ) ;
	 double yw ( 0 ) ;
	 double zw ( 0 ) ;
	 
	 for( DetIdVec::const_iterator j ( detIds.begin() ) ; j != detIds.end() ; ++j ) 
	   {
	     const DetId dId ( (*j).first )  ;
	     const float frac( (*j).second ) ;
	     typename HitTypeCollection::const_iterator iR ( iRecHits->find( dId ) ) ;
	     const double e_j ( iR->energy() *frac) ;
	     
	     double weight = 0;
	     if ( param_LogWeighted_ ) {
	       if ( e_j > 0 ) {
		 weight = std::max( 0., param_W0_ + log(e_j/eTot) );
	       } else {
		 weight = 0;
	       }
	     } else {
	       weight = e_j/eTot;
	     }
	     
	     const CaloCellGeometry* cell ( iSubGeom->getGeometry( dId ) ) ;
	     const float depth ( maxDepth + maxToFront - cell->getPosition().mag() ) ;
	     
	     const GlobalPoint pos (dynamic_cast<const TruncatedPyramid*>( cell )->getPosition( depth ) );
	     
	     xw += weight*pos.x() ;
	     yw += weight*pos.y() ;
	     zw += weight*pos.z() ;
	     
	     total_weight += weight ;
	   }
	 returnValue = math::XYZPoint( xw/total_weight, 
				       yw/total_weight, 
				       zw/total_weight ) ;
       }
   }
  return returnValue ;
}

#endif
