#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/ShashlikNumberingRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/ShashlikDDDConstants.h"
#include "Geometry/CaloTopology/interface/ShashlikTopology.h"
#include "Geometry/FCalGeometry/interface/ShashlikGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

class ShashlikGeometryAnalyzer : public edm::EDAnalyzer
{
public:
  explicit ShashlikGeometryAnalyzer( const edm::ParameterSet& );
  ~ShashlikGeometryAnalyzer( void );
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

private:
  
  void doTest( const ShashlikTopology& topology );
};

ShashlikGeometryAnalyzer::ShashlikGeometryAnalyzer( const edm::ParameterSet& )
{}

ShashlikGeometryAnalyzer::~ShashlikGeometryAnalyzer( void )
{}

void
ShashlikGeometryAnalyzer::analyze( const edm::Event& , 
				   const edm::EventSetup& iSetup )
{
  edm::ESHandle<CaloSubdetectorGeometry> shgeo;
  iSetup.get<ShashlikGeometryRecord>().get( shgeo );
  if(! shgeo.isValid())
    std::cout << "Cannot get a valid ShashlikGeometry Object\n";

  const CaloSubdetectorGeometry* geometry = shgeo.product();
  const std::vector<DetId>& ids = geometry->getValidDetIds();

  edm::ESHandle<ShashlikTopology> topo;
  iSetup.get<ShashlikNumberingRecord>().get(topo);
  const ShashlikTopology* topology = topo.product();
  
  std::vector<int> dins;

  int counter = 0;
  for( std::vector<DetId>::const_iterator i = ids.begin(), iEnd = ids.end(); i != iEnd; ++i, ++counter )
  {
    EKDetId ekid = (*i);
    std::cout << counter << ": din " << topology->detId2denseId(*i) << ":" << ekid << std::endl;
    dins.push_back( topology->detId2denseId(*i));
    
    const CaloCellGeometry * cell = geometry->getGeometry(*i);
    std::cout << *cell << std::endl;
  }
}

void
ShashlikGeometryAnalyzer::doTest( const ShashlikTopology& topology )
{  
  for( int izz = 0; izz <= 1; ++izz )
  {
    int ro(0), fib(0);
    int iz = (2*izz-1);
    for( int ix = 1; ix <= 256; ++ix )
    {
      for( int iy = 1; iy <= 256; ++iy )
      {
	const EKDetId id(ix,iy,fib,ro,iz);
	if( topology.valid( id ))
	{
	  std::cout << "Neighbours for : (" << ix << "," << iy << ") Tower " 
		    << id << std::endl;
	  std::vector<DetId> idE = topology.east(id);
	  std::vector<DetId> idW = topology.west(id);
	  std::vector<DetId> idN = topology.north(id);
	  std::vector<DetId> idS = topology.south(id);
	  std::cout << "          " << idE.size() << " sets along East:";
	  for( unsigned int i = 0; i < idE.size(); ++i ) 
	    std::cout << " " << (EKDetId)(idE[i]());
	  std::cout << std::endl;
	  std::cout << "          " << idW.size() << " sets along West:";
	  for (unsigned int i=0; i<idW.size(); ++i) 
	    std::cout << " " << (EKDetId)(idW[i]());
	  std::cout << std::endl;
	  std::cout << "          " << idN.size() << " sets along North:";
	  for (unsigned int i=0; i<idN.size(); ++i) 
	    std::cout << " " << (EKDetId)(idN[i]());
	  std::cout << std::endl;
	  std::cout << "          " << idS.size() << " sets along South:";
	  for (unsigned int i=0; i<idS.size(); ++i) 
	    std::cout << " " << (EKDetId)(idS[i]());
	  std::cout << std::endl;
	}
      }
    }
  }
}

DEFINE_FWK_MODULE( ShashlikGeometryAnalyzer);
