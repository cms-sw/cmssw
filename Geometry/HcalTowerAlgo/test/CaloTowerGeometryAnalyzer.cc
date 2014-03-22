#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include <fstream>
#include <iostream>
#include <cmath>

namespace 
{
  bool
  AreSame( double a, double b, double epsilon )
  {
    return std::fabs( a - b ) < epsilon;
  }

  inline double
  round( double n, int digits )
  {
    double mult = pow( 10, digits );
    return floor( n * mult ) / mult;
  }
  
  std::map<int, std::string> hcalmapping = {{1, "HB"},
					    {2, "HE"},
					    {3, "HO"},
					    {4, "HF"},
					    {5, "HT"},
					    {6, "ZDC"},
					    {0, "Empty"}
  };
}

class CaloTowerGeometryAnalyzer : public edm::EDAnalyzer 
{
public:
  explicit CaloTowerGeometryAnalyzer( const edm::ParameterSet& );
  ~CaloTowerGeometryAnalyzer( void );
    
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

private:
  std::string m_fname;
  double m_epsilon;
};

CaloTowerGeometryAnalyzer::CaloTowerGeometryAnalyzer( const edm::ParameterSet& iConfig )
  : m_fname( "CaloTower.cells" ),
    m_epsilon( 0.004 )
{
  m_fname = iConfig.getParameter<std::string>( "FileName" );
  m_epsilon = iConfig.getParameter<double>( "Epsilon" );
}

CaloTowerGeometryAnalyzer::~CaloTowerGeometryAnalyzer( void )
{
}

void
CaloTowerGeometryAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
  std::fstream fAll( std::string( m_fname + ".all" ).c_str(), std::ios_base::out );
  std::fstream f( std::string( m_fname + ".diff" ).c_str(), std::ios_base::out );

  edm::ESHandle<CaloGeometry> caloGeom;
  iSetup.get<CaloGeometryRecord>().get( caloGeom );

  const std::vector<DetId>& dha( caloGeom->getSubdetectorGeometry( DetId::Hcal, 1 )->getValidDetIds());

  const std::vector< DetId >& ids (caloGeom->getValidDetIds());

  fAll << std::setw( 4 ) << "iEta"
       << std::setw( 4 ) << "iPhi"
       << std::setw( 4 ) << "Z"
       << std::setw( 10 ) << "Sub:Depth"
       << std::setw( 6 ) << "Eta"
       << std::setw( 13 ) << "Phi"
       << std::setw( 18 ) << "Corners in Eta\n";
  
  f << std::setw( 4 ) << "iEta"
    << std::setw( 4 ) << "iPhi"
    << std::setw( 4 ) << "Z"
    << std::setw( 10 ) << "Sub:Depth"
    << std::setw( 6 ) << "Eta"
    << std::setw( 13 ) << "Phi"
    << std::setw( 18 ) << "Corners in Eta\n";

  for( std::vector<DetId>::const_iterator i( ids.begin()), iEnd( ids.end()) ; i != iEnd; ++i )
  {    
    const CaloGenericDetId cgid( *i );
    if( cgid.isCaloTower()) 
    {
      const CaloTowerDetId cid( *i );
      const int ie( cid.ieta());
      const int ip( cid.iphi());
      const int iz( cid.zside());
      
      fAll << std::setw( 4 ) << ie
	   << std::setw( 4 ) << ip
	   << std::setw( 4 ) << iz
	   << std::setw( 6 ) << "-";
      
      const CaloCellGeometry *cell = caloGeom->getGeometry( *i );
      assert( cell );
      const GlobalPoint& pos = cell->getPosition();
      double eta = pos.eta();
      double phi = pos.phi();
      fAll << std::setw( 10 ) << eta << std::setw( 13 ) << phi;

      const CaloCellGeometry::CornersVec& corners(cell->getCorners());
      for( unsigned int i( 0 ); i != corners.size() ; ++i ) 
      {
	const GlobalPoint& cpos = corners[ i ];
	double ceta = cpos.eta();

	fAll << std::setw( 13 ) << ceta;
      }
      fAll << "\n";
      
      for( std::vector<DetId>::const_iterator ii( dha.begin()), iiEnd( dha.end()); ii != iiEnd; ++ii )
      {
	const HcalDetId hid( *ii );
	const int iie( hid.ieta());
	const int iip( hid.iphi());
	const int iiz( hid.zside());
	const int iid( hid.depth());

	if( ie == iie && ip == iip && iz == iiz ) 
	{
	  fAll << std::setw( 4 ) << iie
	       << std::setw( 4 ) << iip
	       << std::setw( 4 ) << iiz;

	  std::map<int, std::string>::const_iterator iter = hcalmapping.find( hid.subdet());
	  if( iter != hcalmapping.end())
	  {
	    fAll << std::setw( 4 ) << iter->second.c_str() << ":" << iid;
	  }
	  
	  const CaloCellGeometry *hcell = caloGeom->getGeometry( *ii );
	  assert( hcell );
	  const GlobalPoint& hpos = hcell->getPosition();
	  double heta = hpos.eta();
	  double hphi = hpos.phi();
	  
	  fAll << std::setw( 10 ) << heta << std::setw( 13 ) << hphi;

	  const CaloCellGeometry::CornersVec& hcorners(hcell->getCorners());
	  for( unsigned int i( 0 ) ; i != hcorners.size() ; ++i ) 
	  {
	    const GlobalPoint& hcpos = hcorners[ i ];
	    double hceta = hcpos.eta();

	    fAll << std::setw( 13 ) << hceta;
	  }

	  if( !AreSame( eta, heta, m_epsilon ))
	  {
	    
	    fAll << "*DIFFER in Eta*";
	    
	    f << std::setw( 4 ) << ie
	      << std::setw( 4 ) << ip
	      << std::setw( 4 ) << iz
	      << std::setw( 6 ) << "-";
	    
	    f << std::setw( 10 ) << eta << std::setw( 13 ) << phi;
	    
	    for( unsigned int i( 0 ) ; i != corners.size() ; ++i ) 
	    {
	      const GlobalPoint& cpos = corners[ i ];
	      double ceta = cpos.eta();

	      f << std::setw( 9 ) << ceta;
	    }
	    f << "\n";
	    
	    f << std::setw( 4 ) << iie
	      << std::setw( 4 ) << iip
	      << std::setw( 4 ) << iiz;
	    
	    if( iter != hcalmapping.end())
	    {
	      f << std::setw( 4 ) << iter->second.c_str() << ":" << iid;
	    }

	    f << std::setw( 10 ) << heta << std::setw( 13 ) << hphi;

	    for( unsigned int i( 0 ) ; i != hcorners.size() ; ++i ) 
	    {
	      const GlobalPoint& hcpos = hcorners[ i ];
	      double hceta = hcpos.eta();

	      f << std::setw( 9 ) << hceta;
	    }
	    f << "\n\n";
	  }
	  
	  if( !AreSame( phi, hphi, m_epsilon ))
	    fAll << " *DIFFER in Phi*";
	  fAll << "\n";
	}
      }
    }
  }
  
  fAll.close();
  f.close();
}

DEFINE_FWK_MODULE(CaloTowerGeometryAnalyzer);
