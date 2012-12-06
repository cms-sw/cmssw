#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include <iostream>

class HcalGeometryAnalyzer : public edm::EDAnalyzer 
{
public:
    explicit HcalGeometryAnalyzer( const edm::ParameterSet& );
    ~HcalGeometryAnalyzer( void );
    
    virtual void analyze( const edm::Event&, const edm::EventSetup& );

private:
    const HcalFlexiHardcodeGeometryLoader& m_loader;
};

HcalGeometryAnalyzer::HcalGeometryAnalyzer( const edm::ParameterSet& iConfig ) 
    : m_loader( iConfig )
{}

HcalGeometryAnalyzer::~HcalGeometryAnalyzer( void )
{}

void
HcalGeometryAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
    edm::ESHandle<HcalTopology> topologyHandle;
    iSetup.get<IdealGeometryRecord>().get( topologyHandle );
    const HcalTopology* topology ( topologyHandle.product() ) ;

    edm::ESHandle<CaloSubdetectorGeometry> pG;
    iSetup.get<HcalGeometryRecord>().get( HcalGeometry::producerTag() + std::string("_master"), pG );
    
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
    std::cout << "=== Total " << counter << " cells in HCAL." << std::endl;

    counter = 0;
    for (std::vector<int>::const_iterator i=dins.begin(); i != dins.end(); ++i, ++counter)
    {
	HcalDetId hid = (topology->denseId2detId(*i));
	HcalDetId ihid = (topology->denseId2detId(dins[counter]));
	std::cout << counter << ": din " << (*i) << " :" << hid << " == " << ihid << std::endl;
    }    
}

DEFINE_FWK_MODULE(HcalGeometryAnalyzer);
