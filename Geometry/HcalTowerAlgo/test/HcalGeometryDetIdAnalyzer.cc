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

class HcalGeometryDetIdAnalyzer : public edm::EDAnalyzer 
{
public:
    explicit HcalGeometryDetIdAnalyzer( const edm::ParameterSet& );
    ~HcalGeometryDetIdAnalyzer( void );
    
    virtual void analyze( const edm::Event&, const edm::EventSetup& );

private:
    const HcalFlexiHardcodeGeometryLoader& m_loader;
    std::string m_label;
};

HcalGeometryDetIdAnalyzer::HcalGeometryDetIdAnalyzer( const edm::ParameterSet& iConfig ) 
    : m_loader( iConfig ),
      m_label("_master")
{
    m_label = iConfig.getParameter<std::string>( "HCALGeometryLabel" );
}

HcalGeometryDetIdAnalyzer::~HcalGeometryDetIdAnalyzer( void )
{}

void
HcalGeometryDetIdAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
    edm::ESHandle<HcalTopology> topologyHandle;
    iSetup.get<IdealGeometryRecord>().get( topologyHandle );
    const HcalTopology* topology ( topologyHandle.product() ) ;

    edm::ESHandle<CaloSubdetectorGeometry> pG;
    iSetup.get<HcalGeometryRecord>().get( HcalGeometry::producerTag() + m_label, pG );
    
    const CaloSubdetectorGeometry* caloGeom = pG.product();
    const std::vector<DetId>& ids = caloGeom->getValidDetIds();

    int counter = 0;
    for( std::vector<DetId>::const_iterator i = ids.begin(), iEnd = ids.end(); i != iEnd; ++i, ++counter )
    {
	HcalDetId hid = (*i);
	unsigned int did = topology->detId2denseId(*i);
	HcalDetId rhid = topology->denseId2detId(did);
	
	std::cout << counter << ": din " << did << ": " << hid << " == " << rhid << std::endl;
	assert(hid == rhid);
    }
}

DEFINE_FWK_MODULE(HcalGeometryDetIdAnalyzer);
