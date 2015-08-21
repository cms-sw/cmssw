#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include <iostream>

class HcalGeometryAnalyzer : public edm::one::EDAnalyzer<> 
{
public:
  explicit HcalGeometryAnalyzer( const edm::ParameterSet& );
  ~HcalGeometryAnalyzer( void );
    
  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  std::string m_label;
};

HcalGeometryAnalyzer::HcalGeometryAnalyzer( const edm::ParameterSet& iConfig ) 
    : m_label("_master")
{
    m_label = iConfig.getParameter<std::string>( "HCALGeometryLabel" );
}

HcalGeometryAnalyzer::~HcalGeometryAnalyzer( void )
{}

void
HcalGeometryAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
    edm::ESHandle<HcalTopology> topologyHandle;
    iSetup.get<HcalRecNumberingRecord>().get( topologyHandle );
    const HcalTopology* topology ( topologyHandle.product() ) ;

    edm::ESHandle<CaloSubdetectorGeometry> pG;
    iSetup.get<HcalGeometryRecord>().get( HcalGeometry::producerTag() + m_label, pG );
    
    const CaloSubdetectorGeometry* caloGeom = pG.product();
    const std::vector<DetId>& ids = caloGeom->getValidDetIds();

    std::vector<int> dins;
    int counter = 0;
    for( std::vector<DetId>::const_iterator i = ids.begin(), iEnd = ids.end(); i != iEnd; ++i, ++counter )
    {
	HcalDetId hid = (*i);
	std::cout << counter << ": din " << topology->detId2denseId(*i) << ":" << hid;
	dins.push_back( topology->detId2denseId(*i));
	
	const CaloCellGeometry * cell = caloGeom->getGeometry(*i);
	std::cout << *cell << std::endl;
    }

    std::sort( dins.begin(), dins.end());
    std::cout << "=== Total " << counter << " cells in HCAL."
	      << " from HcalTopology ncells " << topology->ncells() << std::endl;

    // HB : 6911: din 16123
    std::cout << "HB Size " << topology->getHBSize()
	      << "\nHE Size " << topology->getHESize()
	      << "\nHO Size " << topology->getHOSize()
	      << "\nHF Size " << topology->getHFSize()
	      << "\nTotal " << topology->getHBSize() + topology->getHESize() + topology->getHOSize() + topology->getHFSize() 
	      << "\n";
    
    counter = 0;
    for (std::vector<int>::const_iterator i=dins.begin(); i != dins.end(); ++i, ++counter)
    {
	HcalDetId hid = (topology->denseId2detId(*i));
	HcalDetId ihid = (topology->denseId2detId(dins[counter]));
	std::cout << counter << ": din " << (*i) << " :" << hid << " == " << ihid << std::endl;
    }
}

DEFINE_FWK_MODULE(HcalGeometryAnalyzer);
