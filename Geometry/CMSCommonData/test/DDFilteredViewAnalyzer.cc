#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class DDFilteredViewAnalyzer : public edm::one::EDAnalyzer<> {
public:

  explicit DDFilteredViewAnalyzer( const edm::ParameterSet& );
  ~DDFilteredViewAnalyzer( void ) {}

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  std::string m_attribute;
  std::string m_value;
  bool m_shouldPrint;
  DDCompOp m_comp;
};

DDFilteredViewAnalyzer::DDFilteredViewAnalyzer( const edm::ParameterSet& pset ) {
  m_attribute = pset.getParameter< std::string >( "attribute" );
  m_value = pset.getParameter< std::string >( "value" );
  
  m_shouldPrint = pset.getUntrackedParameter<bool>("shouldPrint",true);
  if(pset.getUntrackedParameter<bool>("compareNotEquals",true) ) {
    m_comp = DDCompOp::not_equals;
  } else {
    m_comp= DDCompOp::equals;
  }
}

void
DDFilteredViewAnalyzer::analyze( const edm::Event& , 
				 const edm::EventSetup& iSetup ) {
  edm::ESTransientHandle<DDCompactView> cpv;
  iSetup.get<IdealGeometryRecord>().get( cpv );
 
  DDValue val( m_attribute, m_value, 0.0 );
  DDSpecificsFilter filter;
  filter.setCriteria( val,  // name & value of a variable 
  		      m_comp
  		     );
  DDFilteredView fv( *cpv,filter );
  if( fv.firstChild()) {
    std::cout << "Found attribute " << m_attribute.c_str() << " with value " << m_value.c_str() << std::endl;
    bool dodet = true;
    int i = 0;
    while( dodet ) {
      dodet = fv.next();
      if(m_shouldPrint) {
        std::cout << i++ << ": " << fv.logicalPart().name() << std::endl;
      }
    }
  }
  else
    std::cout << "No luck..." << std::endl;
}

DEFINE_FWK_MODULE( DDFilteredViewAnalyzer );
