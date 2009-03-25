#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"

PositionCalc::PositionCalc( std::map<std::string,double> providedParameters ) :
   param_LogWeighted_  ( providedParameters.find("LogWeighted")->second ) ,
   param_T0_barl_      ( providedParameters.find("T0_barl")->second     ) , 
   param_T0_endc_      ( providedParameters.find("T0_endc")->second     ) , 
   param_T0_endcPresh_ ( providedParameters.find("T0_endcPresh")->second ) , 
   param_W0_           ( providedParameters.find("W0")->second           ) ,
   param_X0_           ( providedParameters.find("X0")->second           ) ,
   m_esGeom            ( 0 ) ,
   m_esPlus            ( false ) ,
   m_esMinus           ( false )
{
}

const PositionCalc& PositionCalc::operator=( const PositionCalc& rhs ) 
{
   param_LogWeighted_ = rhs.param_LogWeighted_;
   param_T0_barl_ = rhs.param_T0_barl_;
   param_T0_endc_ = rhs.param_T0_endc_;
   param_T0_endcPresh_ = rhs.param_T0_endcPresh_;
   param_W0_ = rhs.param_W0_;
   param_X0_ = rhs.param_X0_;

   m_esGeom = rhs.m_esGeom ;
   m_esPlus = rhs.m_esPlus ;
   m_esMinus = rhs.m_esMinus ;
   return *this;
}

math::XYZPoint 
PositionCalc::Calculate_Location( const std::vector< std::pair<DetId, float> >&      iDetIds  ,
				  const EcalRecHitCollection*    iRecHits ,
				  const CaloSubdetectorGeometry* iSubGeom ,
				  const CaloSubdetectorGeometry* iESGeom   )
{
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
      typedef std::vector<DetId> DetIdVec ;

      DetIdVec detIds ;
      detIds.reserve( iDetIds.size() ) ;

      double eTot  ( 0 ) ;
      double eMax  ( 0 ) ;
      DetId  maxId ;

      // Check that DetIds are nonzero
      EcalRecHitCollection::const_iterator endRecHits ( iRecHits->end() ) ;
      for( std::vector< std::pair<DetId, float> >::const_iterator n ( iDetIds.begin() ) ; n != iDetIds.end() ; ++n ) 
      {
	 const DetId dId ( (*n).first ) ;
	 if( !dId.null() )
	 {
	    EcalRecHitCollection::const_iterator iHit ( iRecHits->find( dId ) ) ;
	    if( iHit != endRecHits )
	    {
	       detIds.push_back( dId );

	       const double energy ( iHit->energy() ) ;

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
	 LogDebug("ZeroClusterEnergy") << "cluster with 0 energy: " << eTot
				       << ", returning (0,0,0)";
      }
      else
      {
	 // first time or when es geom changes set flags
	 if( 0        != iESGeom &&
	     m_esGeom != iESGeom    )
	 {
	    m_esGeom = iESGeom ;
	    const CaloSubdetectorGeometry::CellCont& cells ( iESGeom->cellGeometries() ) ;
	    for( CaloSubdetectorGeometry::CellCont::const_iterator ic ( cells.begin() ) ;
		 ic != cells.end() && ( (!m_esPlus) || (!m_esMinus) ) ; ++ic )
	    {
	       const double z ( (*ic)->getPosition().z() ) ;
	       m_esPlus  = m_esPlus  || ( 0 < z ) ;
	       m_esMinus = m_esMinus || ( 0 > z ) ;
	    }
	 }

	 //Select the correct value of the T0 parameter depending on subdetector

	 const CaloCellGeometry* center_cell ( iSubGeom->getGeometry( maxId ) ) ;
	 const double ctreta ( center_cell->getPosition().eta() ) ;

	 // for barrel, use barrel T0; 
	 // for endcap: if preshower present && in preshower fiducial, use preshower T0
	 //             else use endcap only T0

	 const Double32_t T0 ( 1.479 > fabs( ctreta ) ? param_T0_barl_ :
			       ( ( ( 1.653 < fabs( ctreta ) ) &&
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
	    const DetId dId ( *j ) ;
	    EcalRecHitCollection::const_iterator iR ( iRecHits->find( dId ) ) ;
	    const double e_j ( iR->energy() ) ;

	    const double weight ( param_LogWeighted_ ? 
				  std::max( 0., param_W0_ + log( e_j/eTot) ) :
				  e_j/eTot ) ;
    
	    const CaloCellGeometry* cell ( iSubGeom->getGeometry( dId ) ) ;

	    const float depth ( maxDepth + maxToFront - cell->getPosition().mag() ) ;

	    const GlobalPoint pos (
	       dynamic_cast<const TruncatedPyramid*>( cell )->getPosition( depth ) );

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
