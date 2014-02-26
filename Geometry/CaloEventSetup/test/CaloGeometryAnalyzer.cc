// -*- C++ -*-
//
// Package:    CaloGeometryAnalyzer
// Class:      CaloGeometryAnalyzer
// 
/**\class CaloGeometryAnalyzer CaloGeometryAnalyzer.cc test/CaloGeometryAnalyzer/src/CaloGeometryAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include "DataFormats/EcalDetId/interface/EcalScDetId.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <fstream>
#include <iomanip>
#include <iterator>
#include "CLHEP/Units/GlobalSystemOfUnits.h"  
#include "TH1.h"
#include "TH1D.h"
#include "TProfile.h"
//
// class decleration


class CaloGeometryAnalyzer : public edm::EDAnalyzer 
{
      enum CenterOrCorner { kCenter , kCorner } ;
      enum XorYorZ { kX, kY, kZ } ;

   public:

      explicit CaloGeometryAnalyzer( const edm::ParameterSet& );
      ~CaloGeometryAnalyzer();

      virtual void analyze( const edm::Event&, const edm::EventSetup& );

   private:
      // ----------member data ---------------------------
      void build( const CaloGeometry& cg , const HcalTopology& ht,
		  DetId::Detector     det, 
		  int                 subdetn, 
		  const char*         name,
		  unsigned int        histi   );

      void buildHcal( const CaloGeometry& cg , const HcalTopology& ht,
		  DetId::Detector     det, 
		  int                 subdetn, 
		  const char*         name,
		  unsigned int        histi   );

      void ctrcor( const DetId&            did     ,
		   const CaloCellGeometry& cell ,
		   std::fstream&           fCtr    ,
		   std::fstream&           fCor    ,  
		   std::fstream&           oldCtr    ,
		   std::fstream&           oldCor   ,
		   unsigned int            histi        );

      void checkDiff( int            i1,
		      int            i2,
		      int            i3,
		      CenterOrCorner iCtrCor ,
		      XorYorZ        iXYZ    ,
		      double         diff      ) ;
      int pass_;
      //  bool fullEcalDump_;

      EEDetId gid( unsigned int ix, 
		   unsigned int iy,
		   unsigned int iz,
		   const EEDetId& did ) const ;

      void cmpset( const CaloSubdetectorGeometry* geom ,
		   const GlobalPoint&             gp   ,
		   const double                   dR     ) ;

      void ovrTst( const CaloGeometry& cg      , 
		   const CaloSubdetectorGeometry* geom ,
		   const EEDetId&   id   , 
		   std::fstream&    fOvr  );

      void ovrTst( const CaloGeometry& cg      , 
		   const CaloSubdetectorGeometry* geom ,
		   const EBDetId&   id   , 
		   std::fstream&    fOvr  );


      edm::Service<TFileService> h_fs;


      TProfile* h_dPhi[7] ;
      TProfile* h_dPhiR[7] ;

      TProfile* h_dEta[7] ;
      TProfile* h_dEtaR[7] ;

      TProfile* h_eta ;
      TProfile* h_phi;

      TH1D* h_diffs[10][12] ;

      TH1D* h_scindex ;

      bool m_allOK ;
};
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CaloGeometryAnalyzer::CaloGeometryAnalyzer( const edm::ParameterSet& /*iConfig*/ )
{
   //now do what ever initialization is needed
  pass_=0;
  //  fullEcalDump_=iConfig.getUntrackedParameter<bool>("fullEcalDump",false);
  h_dPhi[0] = h_fs->make<TProfile>("dPhi:EB:index", "EB: dPhi vs index", 61200, -0.5, 61199.5, " " ) ;
  h_dPhiR[0]= h_fs->make<TProfile>("dPhi:EB:R", "EB: dPhi vs R", 100, 125, 135, " " ) ;

  h_dEta[0] = h_fs->make<TProfile>("dEta:EB:index", "EB: dEta vs index", 61200, -0.5, 61199.5, " " ) ;
  h_dEtaR[0]= h_fs->make<TProfile>("dEta:EB:R", "EB: dEta vs R", 100, 125, 135, " " ) ;

  h_dPhi[1] = h_fs->make<TProfile>("dPhi:EE:index", "EE: dPhi vs index", 14648, -0.5, 14647.5, " " ) ;
  h_dPhiR[1]= h_fs->make<TProfile>("dPhi:EE:R", "EE: dPhi vs R", 130, 30, 160, " " ) ;

  h_dEta[1] = h_fs->make<TProfile>("dEta:EE:index", "EE: dEta vs index", 14648, -0.5, 14647.5, " " ) ;
  h_dEtaR[1]= h_fs->make<TProfile>("dEta:EE:R", "EE: dEta vs R", 130, 30, 160, " " ) ;

  h_dPhi[2] = h_fs->make<TProfile>("dPhi:ES:index", "ES: dPhi vs index", 137216, -0.5, 137215.5, " " ) ;
  h_dPhiR[2]= h_fs->make<TProfile>("dPhi:ES:R", "ES: dPhi vs R", 90, 40, 130, " " ) ;

  h_dEta[2] = h_fs->make<TProfile>("dEta:ES:index", "ES: dEta vs index", 137216, -0.5, 137215.5, " " ) ;
  h_dEtaR[2]= h_fs->make<TProfile>("dEta:ES:R", "ES: dEta vs R", 90, 40, 130, " " ) ;

  h_dPhi[3] = h_fs->make<TProfile>("dPhi:HC:index", "HC: dPhi vs index", 9072, -0.5, 9071.5, " " ) ;
  h_dPhiR[3]= h_fs->make<TProfile>("dPhi:HC:R", "HC: dPhi vs R", 400, 0, 400, " " ) ;

  h_dEta[3] = h_fs->make<TProfile>("dEta:HC:index", "HC: dEta vs index", 9072, -0.5, 9071.5, " " ) ;
  h_dEtaR[3]= h_fs->make<TProfile>("dEta:HC:R", "HC: dEta vs R", 400, 0, 400, " " ) ;

  h_dPhi[4] = h_fs->make<TProfile>("dPhi:ZD:index", "ZD: dPhi vs index", 22, -0.5, 21.5, " " ) ;
  h_dPhiR[4]= h_fs->make<TProfile>("dPhi:ZD:R", "ZD: dPhi vs R", 100, 0, 10, " " ) ;

  h_dEta[4] = h_fs->make<TProfile>("dEta:ZD:index", "ZD: dEta vs index", 22, -0.5, 21.5, " " ) ;
  h_dEtaR[4]= h_fs->make<TProfile>("dEta:ZD:R", "ZD: dEta vs R", 100, 0, 10, " " ) ;

  h_dPhi[5] = h_fs->make<TProfile>("dPhi:CA:index", "CA: dPhi vs index", 224, -0.5, 223.5, " " ) ;
  h_dPhiR[5]= h_fs->make<TProfile>("dPhi:CA:R", "CA: dPhi vs R", 100, 0, 20, " " ) ;

  h_dEta[5] = h_fs->make<TProfile>("dEta:CA:index", "CA: dEta vs index", 224, -0.5, 223.5, " " ) ;
  h_dEtaR[5]= h_fs->make<TProfile>("dEta:CA:R", "CA: dEta vs R", 100, 0, 20, " " ) ;

  h_dPhi[6] = h_fs->make<TProfile>("dPhi:CT:index", "CT: dPhi vs index", 4320, -0.5, 4319.5, " " ) ;
  h_dPhiR[6]= h_fs->make<TProfile>("dPhi:CT:R", "CT: dPhi vs R", 150, 0, 150, " " ) ;

  h_dEta[6] = h_fs->make<TProfile>("dEta:CT:index", "CT: dEta vs index", 4320, -0.5, 4319.5, " " ) ;
  h_dEtaR[6]= h_fs->make<TProfile>("dEta:CT:R", "CT: dEta vs R", 150, 0, 150, " " ) ;

  h_eta = h_fs->make<TProfile>("iEta", "Eta vs iEta", 86*2*4, -86, 86, " " ) ;
  h_phi = h_fs->make<TProfile>("iPhi", "Phi vs iPhi", 360*4, 1, 361, " " ) ;

  const std::string hname[10] = { "EB", "EE", "ES", "HB", "HO", "HE", "HF", "CT", "ZD", "CA" } ;
  const std::string cname[12] = { "XCtr", "YCtr", "ZCtr",
				  "XCor0", "YCor0", "ZCor0",
				  "XCor3", "YCor3", "ZCor3",
				  "XCor6", "YCor6", "ZCor6" } ;

  for( unsigned int i ( 0 ) ; i != 10 ; ++i )
  {
     for( unsigned int j ( 0 ) ; j != 12 ; ++j )
     {
	h_diffs[i][j] = h_fs->make<TH1D>( std::string( hname[i] + cname[j] +
						       std::string("Diff (microns)") ).c_str(), 
					  std::string( hname[i] +
						       std::string(": New-Nom(")
						       + cname[j] + std::string(")") ).c_str(), 
					  200, -200., 200. ) ;
     }
  }
  h_scindex = h_fs->make<TH1D>( std::string( "Supercrystal Hashed Index").c_str(), 
				std::string( "SC Hashed Index").c_str(), 
				632, -0.5, 631.5 ) ;
}


CaloGeometryAnalyzer::~CaloGeometryAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

void 
CaloGeometryAnalyzer::cmpset( const CaloSubdetectorGeometry* geom ,
			      const GlobalPoint&             gp   ,
			      const double                   dR     ) 
{
   typedef CaloSubdetectorGeometry::DetIdSet DetSet ;
//   typedef std::vector< CaloSubdetectorGeometry::DetIdSet::value_type > DetVec ;
   const DetSet base ( geom->CaloSubdetectorGeometry::getCells( gp, dR ) ) ;
   const DetSet over ( geom->getCells( gp, dR ) ) ;
   if( over == base )
   {
/*
      std::cout << "getCells Test dR="
		<< dR
		<< ", gp=" << gp 
		<< ", gp.eta=" << gp.eta() 
		<< ", gp.phi=" << gp.phi() 
		<< ": base and over are equal!\n ***************************\n " 
		<< std::endl ;
*/
   }
   else
   {
      if( 2 < abs( base.size() - over.size() ) ) 
      {
      DetSet inBaseNotOver ;
      DetSet inOverNotBase ;
      std::set_difference( base.begin(), base.end(), 
			   over.begin(), over.end(),
			   std::inserter( inBaseNotOver,
					  inBaseNotOver.begin() )  ) ;

      if( inBaseNotOver.empty() )
      {
	 std::cout << "getCells Test dR="
		   << dR
		   << ", gp=" << gp 
		   << ", gp.eta=" << gp.eta() 
		   << ", gp.phi=" << gp.phi() 
		   << ": No elements in base but not overload " 
		   << std::endl ;
      }
      else
      {
	 std::cout << "Length of Base is "<<base.size() <<std::endl ;
	 std::cout << "Length of Over is "<<over.size() <<std::endl ;
	 std::cout << "There are "<<inBaseNotOver.size()
		   << " items in Base but not in Overload"
		   << std::endl ;

	 for( DetSet::const_iterator iS ( inBaseNotOver.begin() ) ;
	      iS != inBaseNotOver.end() ; ++iS )
	 {
	    std::cout<<"getCells Test dR="
		     <<dR
		     << ", gp=" << gp 
		     << ": cell in base but not overload = " ;
	    if( iS->det() == DetId::Ecal &&
		iS->subdetId() == EcalBarrel ) std::cout << EBDetId( *iS ) ;
	    if( iS->det() == DetId::Ecal &&
		iS->subdetId() == EcalEndcap ) std::cout << EEDetId( *iS ) ;
	    if( iS->det() == DetId::Hcal ) std::cout << HcalDetId( *iS ) ;
	    std::cout<< std::endl ;
	 }
      }

      std::set_difference( over.begin(), over.end(),
			   base.begin(), base.end(), 
			   std::inserter( inOverNotBase,
					  inOverNotBase.begin() ) ) ;

      if( inOverNotBase.empty() )
      {
	 std::cout << "getCells Test dR="
		   << dR
		   << ", gp=" << gp 
		   << ", gp.eta=" << gp.eta() 
		   << ", gp.phi=" << gp.phi() 
		   << ": No elements in overload but not base " 
		   << std::endl ;
      }
      else
      {
	 std::cout << "Length of Base is "<<base.size() <<std::endl ;
	 std::cout << "Length of Over is "<<over.size() <<std::endl ;
	 std::cout << "There are "<< inOverNotBase.size()
		   << " items in Overload but not in Base"
		   << std::endl ;

	 for( DetSet::const_iterator iS ( inOverNotBase.begin() ) ;
	      iS != inOverNotBase.end() ; ++iS )
	 {
	    std::cout<<"getCells Test dR="
		     <<dR
		     << ", gp=" << gp 
		     << ": cell in overload but not base = " ;
	    if( iS->det() == DetId::Ecal &&
		iS->subdetId() == EcalBarrel ) std::cout << EBDetId( *iS ) ;
	    if( iS->det() == DetId::Ecal &&
		iS->subdetId() == EcalEndcap ) std::cout << EEDetId( *iS ) ;
	    if( iS->det() == DetId::Hcal ) std::cout << HcalDetId( *iS ) ;
	    std::cout << std::endl ;
	 }
      }
      std::cout<<"------------- done with mismatch printout ---------------"<<std::endl ;
      }
   }
//   assert( over == base ) ;
}

EEDetId
CaloGeometryAnalyzer::gid( unsigned int ix, 
			   unsigned int iy,
			   unsigned int iz,
			   const EEDetId& did ) const
{
   return ( EEDetId::validDetId( ix, iy, iz ) ? 
	    EEDetId( ix, iy, iz ) : 
	    ( EEDetId::validDetId( ix+1, iy  , iz ) ? 
	      EEDetId( ix+1, iy  , iz ) :
	      ( EEDetId::validDetId( ix-1 , iy, iz ) ? 
		EEDetId( ix-1  , iy, iz ) :
		( EEDetId::validDetId( ix , iy+1, iz ) ? 
		  EEDetId( ix  , iy+1, iz ) :
		  ( EEDetId::validDetId( ix , iy-1, iz ) ? 
		    EEDetId( ix, iy-1, iz ) :
		    ( EEDetId::validDetId( ix+1, iy+1, iz ) ? 
		      EEDetId( ix+1, iy+1, iz ) :
		      ( EEDetId::validDetId( ix+1 , iy-1, iz ) ? 
			EEDetId( ix+1, iy-1, iz ) :
			( EEDetId::validDetId( ix-1, iy+1, iz ) ? 
			  EEDetId( ix-1, iy+1, iz ) :
			  ( EEDetId::validDetId( ix-1, iy-1, iz ) ? 
			    EEDetId( ix-1, iy-1, iz ) : did )  ) ) ) ) ) ) ) ) ;
}

void 
CaloGeometryAnalyzer::ovrTst( const CaloGeometry& cg      , 
			      const CaloSubdetectorGeometry* geom ,
			      const EEDetId&   id   , 
			      std::fstream&    fOvr   )
{
   static const GlobalPoint origin (0,0,0) ;
   const int iphi ( id.iPhiOuterRing() ) ;
   if( iphi != 0 )
   {
      fOvr << "Barrel Neighbors of Endcap id = " << id << std::endl ;
      const EcalEndcapGeometry* eeG ( dynamic_cast<const EcalEndcapGeometry*>( geom ) );
      const CaloCellGeometry* cell ( geom->getGeometry(id) ) ;
      const CaloSubdetectorGeometry* bar ( cg.getSubdetectorGeometry( DetId::Ecal, EcalBarrel ) );
      const EcalEndcapGeometry::OrderedListOfEBDetId* ol ( eeG->getClosestBarrelCells( id ) ) ;
      assert ( 0 != ol ) ;
      for( unsigned int i ( 0 ) ; i != ol->size() ; ++i )
      {
	 fOvr << "           " << i << "  " << (*ol)[i] ;
	 const CaloCellGeometry* other ( bar->getGeometry((*ol)[i]) ) ;
	 const GlobalVector cv ( cell->getPosition()-origin ) ;
	 const GlobalVector ov ( other->getPosition()-origin ) ;
	 const double cosang ( cv.dot(ov)/(cv.mag()*ov.mag() ) ) ;
	 const double angle ( 180.*acos( fabs(cosang)<1?cosang: 1 )/M_PI ) ;
	 fOvr << ", angle = "<<angle<< std::endl ;
      }
   }
}

void 
CaloGeometryAnalyzer::ovrTst( const CaloGeometry& cg      , 
			      const CaloSubdetectorGeometry* geom ,
			      const EBDetId&   id   , 
			      std::fstream&    fOvr   )
{
   static const GlobalPoint origin (0,0,0) ;
   const int ieta ( id.ieta() ) ;
   if( 85 == abs( ieta ) )
   {
      const EcalBarrelGeometry* ebG ( dynamic_cast<const EcalBarrelGeometry*>( geom ) );
      const CaloCellGeometry* cell ( geom->getGeometry(id) ) ;
      const CaloSubdetectorGeometry* ecap ( cg.getSubdetectorGeometry( DetId::Ecal, EcalEndcap ) );
      fOvr << "Endcap Neighbors of Barrel id = " << id << std::endl ;
      const EcalBarrelGeometry::OrderedListOfEEDetId* ol ( ebG->getClosestEndcapCells( id ) ) ;
      assert ( 0 != ol ) ;
      for( unsigned int i ( 0 ) ; i != ol->size() ; ++i )
      {
	 fOvr << "           " << i << "  " << (*ol)[i] ;
	 const CaloCellGeometry* other ( ecap->getGeometry((*ol)[i]) ) ;
	 const GlobalVector cv ( cell->getPosition()-origin ) ;
	 const GlobalVector ov ( other->getPosition()-origin ) ;
	 const double cosang ( cv.dot(ov)/(cv.mag()*ov.mag() ) ) ;
	 const double angle ( 180.*acos( fabs(cosang)<1?cosang: 1 )/M_PI ) ;
	 fOvr << ", angle = "<<angle<< std::endl ;
      }
   }      
}


void 
CaloGeometryAnalyzer::checkDiff( int            i1      ,
				 int            i2      ,
				 int            i3      ,
				 CenterOrCorner iCtrCor ,
				 XorYorZ        iXYZ    ,
				 double         diff      )
{
   if( 3.5 < fabs( diff ) ) 
   {
      std::cout<<"For a volume "<<( kCenter==iCtrCor ? "CENTER" : "CORNER" )
	       <<", & "
	       <<"i1="<<i1<<" & i2="<<i2<<" & i3="<<i3
	       <<", ***BIG DISAGREEMENT FOUND. D"
	       <<( kX == iXYZ ? "X" : ( kY == iXYZ ? "Y" : "Z" ) )
	       <<"=" << diff
	       <<" microns"<<std::endl ;
      m_allOK = false ;
   }
}


void 
CaloGeometryAnalyzer::ctrcor( const DetId&            did     ,
			      const CaloCellGeometry& cell    ,
			      std::fstream&           fCtr    ,
			      std::fstream&           fCor    , 
			      std::fstream&           oldCtr    ,
			      std::fstream&           oldCor    ,
			      unsigned int            histi         )
{
   int oldie ( 0 ) ;
   int oldip ( 0 ) ;
   oldCtr>>oldie>>oldip ;
   oldCor>>oldie>>oldip ;
   const CaloGenericDetId cgid ( did ) ;
   if( cgid.isEB() )
   {
      const EBDetId ebid ( did ) ;
      const int ie  ( ebid.ieta() ) ;
      const int ip  ( ebid.iphi() ) ;
      fCtr << std::setw(4) << ie
	   << std::setw(4) << ip ;
      fCor << std::setw(4) << ie
	   << std::setw(4) << ip ;
   }
   if( cgid.isEE() )
   {
      const EEDetId eeid ( did ) ;
      const int ix ( eeid.ix() ) ;
      const int iy ( eeid.iy() ) ;
//      const int iz ( eeid.zside() ) ;
      fCtr << std::setw(4) << ix
	   << std::setw(4) << iy ;
      fCor << std::setw(4) << ix
	   << std::setw(4) << iy ;
   }
   if( cgid.isES() )
   {
      const ESDetId esid ( did ) ;
      const int pl ( esid.plane() ) ;
      const int ix ( esid.six() ) ;
      const int iy ( esid.siy() ) ;
//      const int iz ( esid.zside() ) ;
      const int st ( esid.strip() ) ;
      fCtr << std::setw(4) << pl
	   << std::setw(4) << ix
	   << std::setw(4) << iy
	   << std::setw(4) << st ;
      fCor << std::setw(4) << pl
	   << std::setw(4) << ix
	   << std::setw(4) << iy
	   << std::setw(4) << st ;
      int oldiy, oldst ;
      oldCtr>>oldiy>>oldst ;
      oldCor>>oldip>>oldst ;
   }
   int depth = 0;
   if( cgid.det() == DetId::Hcal) 
   {
      const HcalDetId hcid ( did ) ;
      const int ie ( hcid.ieta() ) ;
      const int ip ( hcid.iphi() ) ;
      const int de ( hcid.depth() ) ;
//      const int iz ( hcid.zside() ) ;
      fCtr << std::setw(4) << ie
	   << std::setw(4) << ip
	   << std::setw(4) << de ;
      fCor << std::setw(4) << ie
	   << std::setw(4) << ip
	   << std::setw(4) << de ;
      int  oldde ;
      oldCtr>>oldde ;
      oldCor>>oldde ;
      depth = de; 
   }
   if( cgid.isZDC() ) 
   {
      const HcalZDCDetId zcid ( did ) ;
      const int is ( zcid.section() ) ;
      const int ic ( zcid.channel() ) ;
      fCtr << std::setw(4) << is
	   << std::setw(4) << ic ;
      fCor << std::setw(4) << is
	   << std::setw(4) << ic ;
   }
   if( cgid.isCastor() ) 
   {
      const HcalCastorDetId cid ( did ) ;
      const int is ( cid.sector() ) ;
      const int im ( cid.module() ) ;
      fCtr << std::setw(4) << is
	   << std::setw(4) << im ;
      fCor << std::setw(4) << is
	   << std::setw(4) << im ;
   }
   if( cgid.isCaloTower() ) 
   {
      const CaloTowerDetId cid ( did ) ;
      const int ie ( cid.ieta() ) ;
      const int ip ( cid.iphi() ) ;
      fCtr << std::setw(4) << ie
	   << std::setw(4) << ip ;
      fCor << std::setw(4) << ie
	   << std::setw(4) << ip ;
   }

   const double x ( cell.getPosition().x() ) ;
   const double y ( cell.getPosition().y() ) ;
   const double z ( cell.getPosition().z() ) ;

   double oldx,oldy,oldz;

   oldCtr >> oldx >> oldy >> oldz;

   const double dx ( 1.e4*(x - oldx) ) ;
   const double dy ( 1.e4*(y - oldy) ) ;
   const double dz ( 1.e4*(z - oldz) ) ;

   h_diffs[histi][0]->Fill( dx ) ;
   h_diffs[histi][1]->Fill( dy ) ;
   h_diffs[histi][2]->Fill( dz ) ;

   checkDiff( oldie, oldip, depth, kCenter, kX, dx ) ;
   checkDiff( oldie, oldip, depth, kCenter, kY, dy ) ;
   checkDiff( oldie, oldip, depth, kCenter, kZ, dz ) ;

   fCtr << std::fixed << std::setw(12) << std::setprecision(4)
	<< x
	<< std::fixed << std::setw(12) << std::setprecision(4)
	<< y
	<< std::fixed << std::setw(12) << std::setprecision(4)
	<< z
	<< std::endl ;

   const CaloCellGeometry::CornersVec& co ( cell.getCorners() ) ;

   for( unsigned int j ( 0 ) ; j < co.size() ; ++(++(++j)) )
   {
      const double x ( co[j].x() ) ;
      const double y ( co[j].y() ) ;
      const double z ( co[j].z() ) ;

      double oldx,oldy,oldz;

      oldCor >> oldx >> oldy >> oldz;

      const double dx ( 1.e4*(x - oldx) ) ;
      const double dy ( 1.e4*(y - oldy) ) ;
      const double dz ( 1.e4*(z - oldz) ) ;

      h_diffs[histi][j+3]->Fill( dx ) ;
      h_diffs[histi][j+4]->Fill( dy ) ;
      h_diffs[histi][j+5]->Fill( dz ) ;

      checkDiff( oldie, oldip, j, kCorner, kX, dx ) ;
      checkDiff( oldie, oldip, j, kCorner, kY, dy ) ;
      checkDiff( oldie, oldip, j, kCorner, kZ, dz ) ;

      fCor << std::fixed << std::setw(12) << std::setprecision(4)
	   << x
	   << std::fixed << std::setw(12) << std::setprecision(4)
	   << y
	   << std::fixed << std::setw(12) << std::setprecision(4)
	   << z ;
   }
   fCor << std::endl ;
}

void 
CaloGeometryAnalyzer::buildHcal( const CaloGeometry& cg      , 
				 const HcalTopology& ht      ,
				 DetId::Detector     det     , 
				 int                 subdetn , 
				 const char*         name    ,
				 unsigned int        histi     ) 
{
    std::cout<<"Now checking detector "<<name<<std::endl ;
    const std::string oldnameCtr  ( "old" + std::string( name ) + ".ctr" ) ;
    const std::string oldnameCor  ( "old" + std::string( name ) + ".cor" ) ;
    const std::string fnameCtr  ( std::string( name ) + ".ctr" ) ;
    const std::string fnameCor  ( std::string( name ) + ".cor" ) ;
    const std::string fnameOvr  ( std::string( name ) + ".ovr" ) ;
    const std::string fnameRoot ( std::string( name ) + ".C" ) ;
    std::fstream oldCtr(oldnameCtr.c_str() ,std::ios_base::in);
    std::fstream oldCor(oldnameCor.c_str() ,std::ios_base::in);
    std::fstream fCtr(fnameCtr.c_str() ,std::ios_base::out);
    std::fstream fCor(fnameCor.c_str() ,std::ios_base::out);
    std::fstream fOvr(fnameOvr.c_str() ,std::ios_base::out);
    std::fstream f   (fnameRoot.c_str(),std::ios_base::out);

    const CaloSubdetectorGeometry* geom ( cg.getSubdetectorGeometry( det, subdetn ) );

    f << "{" << std::endl;
    f << "  TGeoManager* geoManager = new TGeoManager(\"ROOT\", \"" << name << "\");" << std::endl;
    f << "  TGeoMaterial* dummyMaterial = new TGeoMaterial(\"Vacuum\", 0,0,0); " << std::endl;
    f << "  TGeoMedium* dummyMedium =  new TGeoMedium(\"Vacuum\",1,dummyMaterial);" << std::endl;
    f << "  TGeoVolume* world=geoManager->MakeBox(\"world\",dummyMedium, 8000.0, 8000.0, 14000.0); " << std::endl;
    f << "  geoManager->SetTopVolume(world); " << std::endl;
    f << "  TGeoVolume* box; " << std::endl;
    int n=0;
    const std::vector< DetId >& ids ( geom->getValidDetIds( det, subdetn ) ) ;

    const std::vector< DetId >& ids2 ( cg.getValidDetIds( det, subdetn ) ) ;

    if( ids != ids2 )
    {
	std::cout<<"Methods differ! One gives size "
		 << ids.size()
		 <<" and the other gives size "
		 << ids2.size()
		 << std::endl ;
    }
    assert( ids == ids2 ) ;

    for( std::vector<DetId>::const_iterator i ( ids.begin() ) ; i != ids.end(); ++i ) 
    {
	++n;
	const CaloCellGeometry* cell ( geom->getGeometry(*i) ) ;

	assert( cg.present( *i ) ) ;

	ctrcor( *i,
		*cell,
		fCtr,
		fCor,
		oldCtr,
		oldCor,
		histi ) ;

	const DetId      id ( *i ) ;

	const HcalDetId hcId ( *i ) ;

	const GlobalPoint pos ( cell->getPosition() ) ; 
	const double posmag ( pos.mag() ) ;

	const double disin ( DetId::Ecal   == det     &&
			     EcalPreshower == subdetn     ? 0.000001 : 0.001 ) ;

	const GlobalPoint pointIn ( pos.x() + disin*pos.x()/posmag ,
				    pos.y() + disin*pos.y()/posmag ,
				    pos.z() + disin*pos.z()/posmag   ) ;
	const GlobalPoint pointFr ( pos.x() - 0.1*pos.x()/posmag ,
				    pos.y() - 0.1*pos.y()/posmag ,
				    pos.z() - 0.1*pos.z()/posmag   ) ;

	if( cell->inside( pointFr ) ) std::cout<<"Bad outside: "<<pointIn<<", " <<pointFr<<std::endl ;
	assert( cell->inside( pointIn ) ) ;
	assert( !cell->inside( pointFr ) ) ;

	const double deltaPhi ( geom->deltaPhi( id ) ) ;
      
	const double deltaEta ( geom->deltaEta( id ) ) ;

	const unsigned int detIndex (3);
	const GlobalPoint ggp ( cell->getPosition() ) ;
						
	h_dPhi [detIndex]->Fill( ht.detId2denseId(hcId), deltaPhi ) ;
	h_dPhiR[detIndex]->Fill( ggp.perp(), deltaPhi ) ;

	h_dEta [detIndex]->Fill( ht.detId2denseId(hcId), deltaEta ) ;
	h_dEtaR[detIndex]->Fill( ggp.perp(), deltaEta ) ;
      
	const unsigned int i1 ( HcalGeometry::alignmentTransformIndexLocal( hcId ) ) ;

	const DetId d1 ( HcalGeometry::detIdFromLocalAlignmentIndex( i1 ) ) ;

	const unsigned int i2 ( HcalGeometry::alignmentTransformIndexLocal( d1 ) ) ;

	assert( i1 == i2 ) ;

	f << "  // " << HcalDetId(*i) << std::endl;
	    
	const GlobalPoint gp ( cell->getPosition() ) ;
	f << "  // Checking getClosestCell for position " 
	  << gp
	  << std::endl;

	const HcalDetId closestCell ( geom->getClosestCell( gp ) ) ;

	f << "  // Return position is " << closestCell << std::endl;
	if( closestCell != HcalDetId(*i) )
	{
	    const double rr ( reco::deltaR( gp.eta(), gp.phi(), 
					    geom->getGeometry( closestCell )->getPosition().eta(),
					    geom->getGeometry( closestCell )->getPosition().phi()   ) ) ; 
	    if( rr> 1.e-5 ) std::cout<<"For "<<HcalDetId(*i)<<" closest is "<<closestCell
				     << " HCAL dR=" << rr <<std::endl ;
	}
	// test getCells against base class version every so often
	if( 0 == ht.detId2denseId(closestCell)%30)
	{
	    cmpset( geom, gp,  2*deg ) ;
	    cmpset( geom, gp,  5*deg ) ;
	    cmpset( geom, gp,  7*deg ) ;
	    cmpset( geom, gp, 25*deg ) ;
	    cmpset( geom, gp, 45*deg ) ;
	}
    
	if (det == DetId::Hcal && subdetn==HcalForward) 
	    f << "  box=geoManager->MakeBox(\"point\",dummyMedium,1.0,1.0,1.0);" << std::endl;
	else
	    f << "  box=geoManager->MakeBox(\"point\",dummyMedium,3.0,3.0,3.0);" << std::endl;
	f << "  world->AddNode(box,"<< n << ",new TGeoHMatrix(TGeoTranslation(" << 
	    cell->getPosition().x() << "," << cell->getPosition().y() << "," << cell->getPosition().z() << ")));" << std::endl;
	//   f << (HcalDetId)(*i) << " " << cell->getPosition() << std::endl;
    }
    
    f << "  geoManager->CloseGeometry();" << std::endl;
    f << "world->Voxelize(\"\"); // now the new geometry is valid for tracking, so you can do \n // even raytracing \n //  if (!canvas) { \n    TCanvas* canvas=new TCanvas(\"EvtDisp\",\"EvtDisp\",500,500); \n //  } \n  canvas->Modified(); \n  canvas->Update();      \n  world->Draw(); \n";
    f << "}" << std::endl;
    f.close();
    fCtr.close();
    fCor.close();
}
   


void 
CaloGeometryAnalyzer::build( const CaloGeometry& cg      , 
			     const HcalTopology& ht      ,
			     DetId::Detector     det     , 
			     int                 subdetn , 
			     const char*         name    ,
			     unsigned int        histi     ) 
{
   std::cout<<"Now checking detector "<<name<<std::endl ;

   const std::string oldnameCtr  ( "old" + std::string( name ) + ".ctr" ) ;
   const std::string oldnameCor  ( "old" + std::string( name ) + ".cor" ) ;
   const std::string fnameCtr  ( std::string( name ) + ".ctr" ) ;
   const std::string fnameCor  ( std::string( name ) + ".cor" ) ;
   const std::string fnameOvr  ( std::string( name ) + ".ovr" ) ;
   const std::string fnameRoot ( std::string( name ) + ".C" ) ;
   std::fstream oldCtr(oldnameCtr.c_str() ,std::ios_base::in);
   std::fstream oldCor(oldnameCor.c_str() ,std::ios_base::in);
   std::fstream fCtr(fnameCtr.c_str() ,std::ios_base::out);
   std::fstream fCor(fnameCor.c_str() ,std::ios_base::out);
   std::fstream fOvr(fnameOvr.c_str() ,std::ios_base::out);
   std::fstream f   (fnameRoot.c_str(),std::ios_base::out);

   const CaloSubdetectorGeometry* geom ( cg.getSubdetectorGeometry( det, subdetn ) );

//   std::cout<<"############# parmgr size="<<geom->parMgrConst()->vecSize() << std::endl ;

   f << "{" << std::endl;
   f << "  TGeoManager* geoManager = new TGeoManager(\"ROOT\", \"" << name << "\");" << std::endl;
   f << "  TGeoMaterial* dummyMaterial = new TGeoMaterial(\"Vacuum\", 0,0,0); " << std::endl;
   f << "  TGeoMedium* dummyMedium =  new TGeoMedium(\"Vacuum\",1,dummyMaterial);" << std::endl;
   f << "  TGeoVolume* world=geoManager->MakeBox(\"world\",dummyMedium, 8000.0, 8000.0, 14000.0); " << std::endl;
   f << "  geoManager->SetTopVolume(world); " << std::endl;
   f << "  TGeoVolume* box; " << std::endl;
   int n=0;
   const std::vector< DetId >& ids ( geom->getValidDetIds( det, subdetn ) ) ;

//   std::cout<<"***************total number = "<<ids.size()<<std::endl ;

   const std::vector< DetId >& ids2 ( cg.getValidDetIds( det, subdetn ) ) ;
   
//   std::cout<<"***OTHER METHOD****total number = "<<ids2.size()<<std::endl ;

   if( ids != ids2 )
   {
      std::cout<<"Methods differ! One gives size "
	       << ids.size()
	       <<" and the other gives size "
	       << ids2.size()
	       << std::endl ;
   }

   assert( ids == ids2 ) ;

   for( std::vector<DetId>::const_iterator i ( ids.begin() ) ; i != ids.end(); ++i ) 
   {
      ++n;
      const CaloCellGeometry* cell ( geom->getGeometry(*i) ) ;

      assert( cg.present( *i ) ) ;

      ctrcor( *i,
	      *cell,
	      fCtr,
	      fCor,
	      oldCtr,
	      oldCor,
	      histi ) ;

      const DetId      id ( *i ) ;

      const CaloGenericDetId cid ( id ) ;

      assert( cid.validDetId() ) ;

      assert( CaloGenericDetId( id.det(),
				id.subdetId(),
				cid.denseIndex() ) == id ) ;
      
      const GlobalPoint pos ( cell->getPosition() ) ; 
      const double posmag ( pos.mag() ) ;

      const double disin ( DetId::Ecal   == det     &&
			   EcalPreshower == subdetn     ? 0.000001 : 0.001 ) ;

      const GlobalPoint pointIn ( pos.x() + disin*pos.x()/posmag ,
				  pos.y() + disin*pos.y()/posmag ,
				  pos.z() + disin*pos.z()/posmag   ) ;
      const GlobalPoint pointFr ( pos.x() - 0.1*pos.x()/posmag ,
				  pos.y() - 0.1*pos.y()/posmag ,
				  pos.z() - 0.1*pos.z()/posmag   ) ;

      //assert( cell->inside( pointIn ) ) ;
      if( cell->inside( pointFr ) ) std::cout<<"Bad outside: "<<pointIn<<", " <<pointFr<<std::endl ;
      assert( cell->inside( pointIn ) ) ;
      assert( !cell->inside( pointFr ) ) ;


      const double deltaPhi ( geom->deltaPhi( id ) ) ;
      
      const double deltaEta ( geom->deltaEta( id ) ) ;

      const unsigned int detIndex ( DetId::Ecal == det  && EcalBarrel == subdetn ? 0 :
				    ( DetId::Ecal == det  && EcalEndcap == subdetn ? 1 :
				      ( DetId::Ecal == det  && EcalPreshower == subdetn ? 2 :
					( DetId::Hcal == det                          ? 3 :
					  ( DetId::Calo == det && HcalZDCDetId::SubdetectorId == subdetn ? 4 :
					    ( DetId::Calo == det && HcalCastorDetId::SubdetectorId == subdetn ?
					      5 : 6 )
					     )
					   )
					 )
				       )
	 ) ;

      const CaloGenericDetId cgid ( id ) ;

      const GlobalPoint ggp ( cell->getPosition() ) ;
						
      h_dPhi [detIndex]->Fill( cgid.denseIndex(), deltaPhi ) ;
      h_dPhiR[detIndex]->Fill( ggp.perp(), deltaPhi ) ;

      h_dEta [detIndex]->Fill( cgid.denseIndex(), deltaEta ) ;
      h_dEtaR[detIndex]->Fill( ggp.perp(), deltaEta ) ;



      if( det == DetId::Ecal )
      {
	 if (subdetn == EcalBarrel )
	 {
	    f << "  // " << EBDetId(*i) << std::endl;
	    
	    const GlobalPoint gp ( dynamic_cast<const TruncatedPyramid*>(cell)->getPosition(0.) ) ;

	    f << "  // Checking getClosestCell for position " 
	      << gp
	      << std::endl;


	    const EBDetId ebid ( id ) ;
	    for(unsigned int j ( 0 ) ; j !=4 ; ++j )
	    {
	       const CaloCellGeometry::CornersVec& corn ( cell->getCorners() ) ;
	       h_eta->Fill( ebid.ieta()*1. + 0.25*j, corn[j].eta() ) ;
	       if( ebid.ieta()>0) h_phi->Fill( ebid.iphi()*1. + 0.25*j, corn[j].phi() ) ;
	    }

	    EBDetId closestCell ( geom->getClosestCell( gp ) ) ;

	    f << "  // Return position is " << closestCell << std::endl;
	    assert( closestCell == EBDetId(*i) );
	    // test getCells against base class version every so often
	    if( 0 == closestCell.hashedIndex()%100 )
	    {
	       cmpset( geom, gp,  2*deg ) ;
	       cmpset( geom, gp,  5*deg ) ;
	       cmpset( geom, gp, 25*deg ) ;
	       cmpset( geom, gp, 45*deg ) ;
	    }

	    ovrTst( cg, geom, EBDetId(*i) , fOvr ) ;


	    const unsigned int i1 ( EcalBarrelGeometry::alignmentTransformIndexLocal( ebid ) ) ;

	    const DetId d1 ( EcalBarrelGeometry::detIdFromLocalAlignmentIndex( i1 ) ) ;

	    const unsigned int i2 ( EcalBarrelGeometry::alignmentTransformIndexLocal( d1 ) ) ;

	    assert( i1 == i2 ) ;
	 }
	 if (subdetn == EcalEndcap)
	 {
	    const EEDetId did ( *i ) ;
	    const int ix ( did.ix() ) ;
	    const int iy ( did.iy() ) ;
	    const int iz ( did.zside() ) ;


/*	    const int isc ( did.isc() ) ;
	    //std::cout<<"ix, iy="<<ix<<", "<<iy<<std::endl;

	    const EcalScDetId scId ( 1+(ix-1)/5,1+(iy-1)/5,iz ) ;
	    const int isc2 ( scId.isc() ) ;
	    const int isc2c ( scId.hashedIndex()+1 ) ;

	    h_scindex->Fill( isc2c ) ;

	    int isc3 ( -1 ) ;
	    int icr ( 0 ) ;
	    do
	    {
	       const int jx ( icr%5 + 1 ) ;
	       const int jy ( icr/5 + 1 ) ;
	       if( EEDetId::validDetId( ( scId.ix() - 1 )*5 + jx, 
					( scId.iy() - 1 )*5 + jy, 
					scId.zside() ) )
	       {
		  isc3 = EEDetId( ( scId.ix() - 1 )*5 + jx, 
				  ( scId.iy() - 1 )*5 + jy, 
				  scId.zside() ).isc() ;
//		     + (scId.zside()+1)*316/2 ;
	       }
	       ++icr ;
	    }
	    while( icr<25 && isc3<0 ) ;
	    //if( icr>1 ) std::cout<<"**************** Took extra tries: "<<icr-1<<std::endl;


	    if( isc3 != isc2 )
	    {
	       std::cout<<"**** Supercrystal numbering conflict: isc3, isc2, ix, iy, zside = "
			<<isc3 <<", "<<isc2 <<", "<<ix <<", "<<iy <<", "<<iz <<std::endl ;
	    }
//	    assert( isc3 == isc2 ) ;

	    if( isc + (iz+1)*316/2 != isc2c )
	    {
	       std::cout<<"**** Supercrystal numbering conflict: isc, isc2, ix, iy, zside = "
			<<isc <<", "<<isc2 <<", "<<ix <<", "<<iy <<", "<<iz <<std::endl ;
	    }
	    if( isc == 70 ) std::cout<<"********** Seeing SC=70"<<std::endl ;
*/

	    const unsigned int i1 ( EcalEndcapGeometry::alignmentTransformIndexLocal( did ) ) ;

	    const DetId d1 ( EcalEndcapGeometry::detIdFromLocalAlignmentIndex( i1 ) ) ;

	    const unsigned int i2 ( EcalEndcapGeometry::alignmentTransformIndexLocal( d1 ) ) ;

	    assert( i1 == i2 ) ;


	    const TruncatedPyramid* tp ( dynamic_cast<const TruncatedPyramid*>(cell) ) ;
	    f << "  // Checking getClosestCell for position " << tp->getPosition(0.) << std::endl;

	    const GlobalPoint gp ( tp->getPosition(0.) ) ;

	    const EEDetId closestCell ( geom->getClosestCell( gp ) ) ;
	    f << "  // Return position is " << closestCell << std::endl;

//	    if( closestCell != did ) std::cout<<"eeid = "<<did<<", closest="<<closestCell<<std::endl ;

	    assert( closestCell == did ) ;
	    // test getCells against base class version every so often
	    if( 0 == closestCell.hashedIndex()%10 )
	    {
	       cmpset( geom, gp,  2*deg ) ;
	       cmpset( geom, gp,  5*deg ) ;
	       cmpset( geom, gp, 25*deg ) ;
	       cmpset( geom, gp, 45*deg ) ;
	    }

	    const GlobalVector xx ( 2.5,   0, 0 ) ;
	    const GlobalVector yy (   0, 2.5, 0 ) ;
	    const GlobalVector zz (   0,   0, 1 ) ;
	    const GlobalPoint pointIn ( tp->getPosition(  1.) ) ; 
	    const GlobalPoint pointFr ( tp->getPosition( -1.) ) ; 
	    const GlobalPoint pointBk ( tp->getPosition( 24.) ) ; 
	    const GlobalPoint pointXP ( tp->getPosition(1.) + xx ) ; 
	    const GlobalPoint pointXM ( tp->getPosition(1.) - xx ) ; 
	    const GlobalPoint pointYP ( tp->getPosition(1.) + yy ) ; 
	    const GlobalPoint pointYM ( tp->getPosition(1.) - yy ) ; 
	    const GlobalPoint pointPP ( tp->getPosition(1.) + xx + yy ) ; 
	    const GlobalPoint pointPM ( tp->getPosition(1.) + xx - yy ) ; 
	    const GlobalPoint pointMP ( tp->getPosition(1.) - xx + yy ) ; 
	    const GlobalPoint pointMM ( tp->getPosition(1.) - xx - yy ) ; 
	    const EEDetId didXP ( gid( ix+1, iy  , iz, did ) ) ;
	    const EEDetId didXM ( gid( ix-1, iy  , iz, did ) ) ;
	    const EEDetId didYP ( gid( ix  , iy+1, iz, did ) ) ;
	    const EEDetId didYM ( gid( ix  , iy-1, iz, did ) ) ;
	    const EEDetId didPP ( gid( ix+1, iy+1, iz, did ) ) ;
	    const EEDetId didPM ( gid( ix+1, iy-1, iz, did ) ) ;
	    const EEDetId didMP ( gid( ix-1, iy+1, iz, did ) ) ;
	    const EEDetId didMM ( gid( ix-1, iy-1, iz, did ) ) ;

	    assert(  cell->inside( pointIn ) ) ;
	    assert( !cell->inside( pointFr ) ) ;
	    assert( !cell->inside( pointBk ) ) ;
	    assert( !cell->inside( pointXP ) ) ;
	    assert( !cell->inside( pointXM ) ) ;
	    assert( !cell->inside( pointYP ) ) ;
	    assert( !cell->inside( pointYM ) ) ;
	    assert( !cell->inside( pointPP ) ) ;
	    assert( !cell->inside( pointPM ) ) ;
	    assert( !cell->inside( pointMP ) ) ;
	    assert( !cell->inside( pointMM ) ) ;

	    const EEDetId ccBk ( geom->getClosestCell( pointBk ) ) ;
	    const EEDetId ccIn ( geom->getClosestCell( pointIn ) ) ;
	    const EEDetId ccFr ( geom->getClosestCell( pointFr ) ) ;
	    const EEDetId ccXP ( geom->getClosestCell( pointXP ) ) ;
	    const EEDetId ccXM ( geom->getClosestCell( pointXM ) ) ;
	    const EEDetId ccYP ( geom->getClosestCell( pointYP ) ) ;
	    const EEDetId ccYM ( geom->getClosestCell( pointYM ) ) ;
	    const EEDetId ccPP ( geom->getClosestCell( pointPP ) ) ;
	    const EEDetId ccPM ( geom->getClosestCell( pointPM ) ) ;
	    const EEDetId ccMP ( geom->getClosestCell( pointMP ) ) ;
	    const EEDetId ccMM ( geom->getClosestCell( pointMM ) ) ;

	    assert( ccIn == did ) ;
	    assert( ccFr == did ) ;

//	    if( ccBk != did ) std::cout<<"**eeid="<<did<<", ccBk="<<ccBk<<std::endl;

	    assert( ccBk == did ) ;
	    assert( ccXP == didXP ||
		    !geom->getGeometry(didXP)->inside( pointXP ) ) ;
	    assert( ccXM == didXM ||
		    !geom->getGeometry(didXM)->inside( pointXM ) ) ;
	    assert( ccYP == didYP  ||
		    !geom->getGeometry(didYP)->inside( pointYP ) ) ;
	    assert( ccYM == didYM  ||
		    !geom->getGeometry(didYM)->inside( pointYM )) ;
	    assert( ccPP == didPP  ||
		    !geom->getGeometry(didPP)->inside( pointPP ) ) ;
	    assert( ccPM == didPM ||
		    !geom->getGeometry(didPM)->inside( pointPM ) ) ;
	    assert( ccMP == didMP ||
		    !geom->getGeometry(didMP)->inside( pointMP ) ) ;
	    assert( ccMM == didMM ||
		    !geom->getGeometry(didMM)->inside( pointMM ) ) ;


	    ovrTst( cg, geom, EEDetId(*i) , fOvr ) ;
	 }
	 if (subdetn == EcalPreshower) 
	 {
	    const ESDetId esid ( *i ) ;

	    f << "  // " << esid << std::endl;
	    f << "  // Checking getClosestCell for position " << cell->getPosition() << " in plane " << esid.plane() << std::endl;
	    ESDetId closestCell=ESDetId((dynamic_cast<const EcalPreshowerGeometry*>(geom))->getClosestCellInPlane(cell->getPosition(),esid.plane()));
	    f << "  // Return position is " << closestCell << std::endl;
	    //sanity checks
            int o_zside = esid.zside();
            //int o_plane = ESDetId(*i).plane();
//            int o_six   = esid.six();
//            int o_siy   = esid.siy();
            //int o_strip = ESDetId(*i).strip();

//            assert ((o_six <= 20 && cell->getPosition().x() < 0.) || (o_six > 20 && cell->getPosition().x() > 0.));
//            assert ((o_siy <= 20 && cell->getPosition().y() < 0.) || (o_siy > 20 && cell->getPosition().y() > 0.));
            assert ((o_zside < 0 && cell->getPosition().z() < 0.) ||
		    (o_zside > 0 && cell->getPosition().z() > 0.)        );

	    if( closestCell != esid ) std::cout<<"** esid="<<esid<<", closest="<<closestCell<<std::endl ;

//	    assert (closestCell == esid );

	    
	    const unsigned int i1 ( EcalPreshowerGeometry::alignmentTransformIndexLocal( esid ) ) ;

	    const DetId d1 ( EcalPreshowerGeometry::detIdFromLocalAlignmentIndex( i1 ) ) ;

	    const unsigned int i2 ( EcalPreshowerGeometry::alignmentTransformIndexLocal( d1 ) ) ;

	    assert( i1 == i2 ) ;
	 }
      }
      else if (det == DetId::Hcal)
      {
	 const HcalDetId hcId ( *i ) ;

	 const unsigned int i1 ( HcalGeometry::alignmentTransformIndexLocal( hcId ) ) ;

	 const DetId d1 ( HcalGeometry::detIdFromLocalAlignmentIndex( i1 ) ) ;

	 const unsigned int i2 ( HcalGeometry::alignmentTransformIndexLocal( d1 ) ) ;

	 assert( i1 == i2 ) ;

	 f << "  // " << HcalDetId(*i) << std::endl;
	    
	 const GlobalPoint gp ( cell->getPosition() ) ;

	 f << "  // Checking getClosestCell for position " 
	   << gp
	   << std::endl;

	 const HcalDetId closestCell ( geom->getClosestCell( gp ) ) ;

	 f << "  // Return position is " << closestCell << std::endl;
	 if( closestCell != HcalDetId(*i) )
	 {
	    const double rr ( reco::deltaR( gp.eta(), gp.phi(), 
					    geom->getGeometry( closestCell )->getPosition().eta(),
					    geom->getGeometry( closestCell )->getPosition().phi()   ) ) ; 
	    if( rr> 1.e-5 ) std::cout<<"For "<<HcalDetId(*i)<<" closest is "<<closestCell
				     << " HCAL dR=" << rr <<std::endl ;
	 }
	 // test getCells against base class version every so often
	 if( 0 == ht.detId2denseId(closestCell)%30)
	 {
	    cmpset( geom, gp,  2*deg ) ;
	    cmpset( geom, gp,  5*deg ) ;
	    cmpset( geom, gp,  7*deg ) ;
	    cmpset( geom, gp, 25*deg ) ;
	    cmpset( geom, gp, 45*deg ) ;
	 }
      }
      else if (det == DetId::Calo &&
	       subdetn == HcalCastorDetId::SubdetectorId )
      {
	 f << "  // " << HcalCastorDetId(*i) << std::endl;
	    
	 const GlobalPoint gp ( cell->getPosition().x(),
				cell->getPosition().y(),
				cell->getPosition().z() - 0.1 ) ;

	 f << "  // Checking getClosestCell for position " 
	   << gp
	   << std::endl;

	 const DetId closestCell ( geom->getClosestCell( gp ) ) ;

	 if( closestCell != DetId(0) )
	 {
	    f << "  // Return position is " << HcalCastorDetId(closestCell) << std::endl;
	    if( closestCell != HcalCastorDetId(*i) )
	    {
	       const double rr ( reco::deltaR( gp.eta(), gp.phi(), 
					       geom->getGeometry( closestCell )->getPosition().eta(),
					       geom->getGeometry( closestCell )->getPosition().phi()   ) ) ; 
	       if( rr> 1.e-5 ) std::cout<<"For "<<HcalCastorDetId(*i)<<" closest is "<<HcalCastorDetId(closestCell)
					<< " dR=" << rr <<std::endl ;
	    }
	 }
	 // test getCells against base class version every so often
//	 if( 0 == closestCell.denseIndex()%30 )
	 {
	    cmpset( geom, gp,  2*deg ) ;
	    cmpset( geom, gp,  5*deg ) ;
	    cmpset( geom, gp,  7*deg ) ;
	    cmpset( geom, gp, 25*deg ) ;
	    cmpset( geom, gp, 45*deg ) ;
	 }
      }
      else if (det == DetId::Calo &&
	       subdetn == HcalZDCDetId::SubdetectorId )
      {
	 f << "  // " << HcalZDCDetId(*i) << std::endl;
	 const double sign ( HcalZDCDetId(*i).zside() ) ;
	 const GlobalPoint gp ( cell->getPosition().x(),
				cell->getPosition().y(),
				cell->getPosition().z() + sign*0.1 ) ;

	 f << "  // Checking getClosestCell for position " 
	   << gp
	   << std::endl;

	 const DetId closestCell ( geom->getClosestCell( gp ) ) ;

	 if( closestCell != DetId(0) )
	 {
	    f << "  // Return position is " << HcalZDCDetId(closestCell) << std::endl;
	    if( closestCell != HcalZDCDetId(*i) )
	    {
	       const double rr ( reco::deltaR( gp.eta(), gp.phi(), 
					       geom->getGeometry( closestCell )->getPosition().eta(),
					       geom->getGeometry( closestCell )->getPosition().phi()   ) ) ; 
	       if( rr> 1.e-5 ) std::cout<<"For "<<HcalZDCDetId(*i)<<" closest is "<<HcalZDCDetId(closestCell)
					<< " dR=" << rr <<std::endl ;
	    }
	 }
	 // test getCells against base class version every so often
//	 if( 0 == closestCell.denseIndex()%30 )
	 {
	    cmpset( geom, gp,  2*deg ) ;
	    cmpset( geom, gp,  5*deg ) ;
	    cmpset( geom, gp,  7*deg ) ;
	    cmpset( geom, gp, 25*deg ) ;
	    cmpset( geom, gp, 45*deg ) ;
	 }
      }
    
      if (det == DetId::Hcal && subdetn==HcalForward) 
	 f << "  box=geoManager->MakeBox(\"point\",dummyMedium,1.0,1.0,1.0);" << std::endl;
      else
	 f << "  box=geoManager->MakeBox(\"point\",dummyMedium,3.0,3.0,3.0);" << std::endl;
      f << "  world->AddNode(box,"<< n << ",new TGeoHMatrix(TGeoTranslation(" << 
	 cell->getPosition().x() << "," << cell->getPosition().y() << "," << cell->getPosition().z() << ")));" << std::endl;
      //   f << (HcalDetId)(*i) << " " << cell->getPosition() << std::endl;
   }
   f << "  geoManager->CloseGeometry();" << std::endl;
   f << "world->Voxelize(\"\"); // now the new geometry is valid for tracking, so you can do \n // even raytracing \n //  if (!canvas) { \n    TCanvas* canvas=new TCanvas(\"EvtDisp\",\"EvtDisp\",500,500); \n //  } \n  canvas->Modified(); \n  canvas->Update();      \n  world->Draw(); \n";
   f << "}" << std::endl;
   f.close();
   fCtr.close();
   fCor.close();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
CaloGeometryAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);     
   edm::ESHandle<HcalTopology> pT;
   iSetup.get<HcalRecNumberingRecord>().get(pT);     

   const std::vector<DetId> allDetId ( pG->getValidDetIds() ) ;
//   std::cout<<"Total number of DetIds in all subdets is = "<<allDetId.size()<<std::endl ;

   const std::vector<DetId>& deb ( pG->getValidDetIds(DetId::Ecal,EcalBarrel                     ));
   const std::vector<DetId>& dee ( pG->getValidDetIds(DetId::Ecal,EcalEndcap                     ));
   const std::vector<DetId>& des ( pG->getValidDetIds(DetId::Ecal,EcalPreshower                  ));
   const std::vector<DetId>& dhb ( pG->getValidDetIds(DetId::Hcal,HcalBarrel                     ));
   const std::vector<DetId>& dhe ( pG->getValidDetIds(DetId::Hcal,HcalEndcap                     ));
   const std::vector<DetId>& dho ( pG->getValidDetIds(DetId::Hcal,HcalOuter                      ));
   const std::vector<DetId>& dhf ( pG->getValidDetIds(DetId::Hcal,HcalForward                    ));
   const std::vector<DetId>& dct ( pG->getValidDetIds(DetId::Calo,CaloTowerDetId::SubdetId       ));
   const std::vector<DetId>& dca ( pG->getValidDetIds(DetId::Calo,HcalCastorDetId::SubdetectorId ));
   const std::vector<DetId>& dzd ( pG->getValidDetIds(DetId::Calo,HcalZDCDetId::SubdetectorId    ));

   const std::vector<DetId>& dha ( pG->getSubdetectorGeometry(DetId::Hcal,1)->getValidDetIds());


   const unsigned int sum ( deb.size() +
			    dee.size() +
			    des.size() +
			    dhb.size() +
			    dhe.size() +
			    dho.size() +
			    dhf.size() +
			    dct.size() +
			    dca.size() +
			    dzd.size()   ) ;


//   std::cout<<"Sum in all subdets is = "<<sum<<std::endl ;

   if( sum != allDetId.size() ) 
   {
      std::cout<<"Sums differ! One is "<<allDetId.size()
	       <<" and the other is "<<sum
	       <<std::endl ;
   }

   assert( sum == allDetId.size() ) ;

   assert( dha.size() == dhb.size()+dhe.size()+dho.size()+dhf.size() ) ;
   //
   // get the ecal & hcal geometry
   //
   if (pass_==0) 
   {

      std::cout<<"**Ecal Barrel avg Radius = "
	       << dynamic_cast<const EcalBarrelGeometry*>( pG->getSubdetectorGeometry(DetId::Ecal, EcalBarrel ))->avgRadiusXYFrontFaceCenter()
	       << std::endl ;

      std::cout<<"**Ecal Endcap avg Zabs = "
	       << dynamic_cast<const EcalEndcapGeometry*>( pG->getSubdetectorGeometry(DetId::Ecal, EcalEndcap ))->avgAbsZFrontFaceCenter()
	       << std::endl ;

      m_allOK = true ;

      build(*pG,*pT,DetId::Ecal,EcalBarrel                     ,"eb",0);
      build(*pG,*pT,DetId::Ecal,EcalEndcap                     ,"ee",1);
      build(*pG,*pT,DetId::Ecal,EcalPreshower                  ,"es",2);
      buildHcal(*pG,*pT,DetId::Hcal,HcalBarrel                     ,"hb",3);
      buildHcal(*pG,*pT,DetId::Hcal,HcalEndcap                     ,"he",4);
      buildHcal(*pG,*pT,DetId::Hcal,HcalOuter                      ,"ho",5);
      buildHcal(*pG,*pT,DetId::Hcal,HcalForward                    ,"hf",6);
      build(*pG,*pT,DetId::Calo,CaloTowerDetId::SubdetId       ,"ct",7);
      build(*pG,*pT,DetId::Calo,HcalCastorDetId::SubdetectorId ,"ca",8);
      build(*pG,*pT,DetId::Calo,HcalZDCDetId::SubdetectorId    ,"zd",9);
     //Test eeGetClosestCell in Florian Point
//      std::cout << "Checking getClosestCell for position" << GlobalPoint(-38.9692,-27.5548,-317) << std::endl;
//      std::cout << "Position of Closest Cell in EE " << dynamic_cast<const TruncatedPyramid*>(pG->getGeometry(EEDetId((*pG).getSubdetectorGeometry(DetId::Ecal,EcalEndcap)->getClosestCell(GlobalPoint(-38.9692,-27.5548,-317)))))->getPosition(0.) << std::endl;

      std::cout<<"\n\n*********** Validation of cell centers and corners "
	       <<( m_allOK ? "SUCCEEDS!! " : "FAILS!! ")
	       <<"**********************\n\n\n" 
	       <<std::endl ;
   }

   pass_++;
      
}

//define this as a plug-in

DEFINE_FWK_MODULE(CaloGeometryAnalyzer);
