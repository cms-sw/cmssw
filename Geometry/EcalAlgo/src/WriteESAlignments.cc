#include "Geometry/EcalAlgo/interface/WriteESAlignments.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

typedef WriteESAlignments WEA ;

const unsigned int WEA::k_nA = EcalPreshowerGeometry::numberOfAlignments() ;

WEA::~WriteESAlignments(){}

WEA::WriteESAlignments( const edm::EventSetup& eventSetup ,
			WEA::DVec              alphaVec   ,
			WEA::DVec              betaVec    ,
			WEA::DVec              gammaVec   ,
			WEA::DVec              xtranslVec ,
			WEA::DVec              ytranslVec ,
			WEA::DVec              ztranslVec  ) 
{
   assert( alphaVec.size()   == k_nA ) ;
   assert( betaVec.size()    == k_nA ) ;
   assert( gammaVec.size()   == k_nA ) ;
   assert( xtranslVec.size() == k_nA ) ;
   assert( ytranslVec.size() == k_nA ) ;
   assert( ztranslVec.size() == k_nA ) ;

   AliPtr aliPtr ( new Alignments  ) ;
   AliVec vali   ( aliPtr->m_align ) ;

   convert( eventSetup , 
	    alphaVec   ,
	    betaVec    , 
	    gammaVec   ,
	    xtranslVec , 
	    ytranslVec , 
	    ztranslVec ,
	    vali         ) ;

   write( aliPtr ) ;
}

void 
WEA::write( WEA::AliPtr aliPtr ) 
{
   std::cout << "Uploading ES alignments to the database" << std::endl ;

   edm::Service<cond::service::PoolDBOutputService> poolDbService ;

   if (!poolDbService.isAvailable()) throw cms::Exception("NotAvailable") 
      << "PoolDBOutputService not available";

   poolDbService->writeOne<Alignments>(&(*aliPtr), 
				       poolDbService->currentTime(),
				       "ESAlignmentRcd" ) ;
}

void 
WEA::convert( const edm::EventSetup& eS ,
	      WEA::DVec              a  ,
	      WEA::DVec              b  ,
	      WEA::DVec              g  ,
	      WEA::DVec              x  ,
	      WEA::DVec              y  ,
	      WEA::DVec              z  ,
	      WEA::AliVec&           va   ) 
{
   edm::ESHandle<CaloGeometry>       pG   ;
   eS.get<CaloGeometryRecord>().get( pG ) ;

   const CaloSubdetectorGeometry* geom ( 
      pG->getSubdetectorGeometry( DetId::Ecal, EcalPreshower ) ) ;

   edm::ESHandle<Alignments>  pA ;
   eS.get<ESAlignmentRcd>().get( pA ) ;
   const AliVec vaPrev ( pA->m_align ) ;

   va.reserve( k_nA ) ;
   for( unsigned int i ( 0 ) ; i != k_nA ; ++i )
   {
      // ordering of i is: left, right, left, right,...
      // starting at ES- rear, then ES- front, 
      // then ES+ front, then ES+ rear

      const ESDetId id ( EcalPreshowerGeometry::detIdFromLocalAlignmentIndex( i ) ) ;

      const double zPlanePrev ( geom->getGeometry( id )->getPosition().z() ) ;
      const double zAlignPrev ( vaPrev[i].translation().z() ) ;
      const Trl    q_I ( 0, 0, zPlanePrev - zAlignPrev ) ;
      const Trl&   s_p ( vaPrev[i].translation() ) ;
      const Trl    t_n ( x[i], y[i], z[i] ) ;
      const Rot    G_p ( vaPrev[i].rotation() ) ;
      const double gamma ( g[i] ) ;
      const double alpha ( a[i] ) ;
      const double beta  ( b[i] ) ;

      const Rot L_n ( // New rotation in local frame!
	 Rot( Rot( Rot().rotateZ( -gamma ) ).rotateX( -alpha ) ).rotateY( -beta ) ) ;

      const Rot InvL_n ( L_n.inverse() ) ;

      const Rot G_n ( InvL_n * G_p ) ;

      const Trl s_n ( t_n + s_p + q_I - InvL_n*q_I ) ;

      std::cout<<"For i = "<<i<<", q_I="<<q_I<<std::endl ;
      std::cout<<"For i = "<<i<<", s_p="<<s_p<<std::endl ;
      std::cout<<"For i = "<<i<<", alpha = "<<1000.*alpha<<" mr"<<std::endl;
      std::cout<<"For i = "<<i<<", beta  = "<<1000.*beta <<" mr"<<std::endl;
      std::cout<<"For i = "<<i<<", gamma = "<<1000.*gamma<<" mr"<<std::endl;
      std::cout<<" For i = "<<i<<", L_n = "<< L_n
	       <<"   Euler angles="<<InvL_n.eulerAngles()<<"\n"<<std::endl;
      std::cout<<"For i = "<<i<<", t_n="<<t_n<<std::endl ;
      std::cout<<"For i = "<<i<<", G_p="<<G_p
	       <<"   Euler angles="<<G_p.eulerAngles()<<"\n"<<std::endl ;
      std::cout<<" For i = "<<i<<", InvL_n = "<< InvL_n
	       <<"   Euler angles="<<InvL_n.eulerAngles()<<"\n"<<std::endl;
      std::cout<<" For i ="<<i<<", G_n = "<< G_n
	       <<"    Euler angles="<<G_n.eulerAngles()<<"\n"<<std::endl;
      std::cout<<" For i ="<<i<<", s_n = "<< s_n<<std::endl;
      std::cout<<"++++++++++++++++++++++++++\n\n"<<std::endl;

      va.push_back( AlignTransform( s_n, G_n, id ) ) ;
   }
}
