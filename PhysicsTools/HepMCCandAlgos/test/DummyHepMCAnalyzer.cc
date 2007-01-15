#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>
using namespace std;
using namespace edm;
using namespace HepMC;

class DummyHepMCAnalyzer : public EDAnalyzer {
public:
  explicit DummyHepMCAnalyzer( const ParameterSet & cfg ) : 
    src_( cfg.getParameter<InputTag>( "src" ) ) {
  }
private:
  void analyze( const Event & evt, const EventSetup & ) {
    Handle<HepMCProduct> hepMC;
    evt.getByLabel( src_, hepMC );
    const GenEvent * mc = hepMC->GetEvent();
    if( mc == 0 ) 
      throw edm::Exception( edm::errors::InvalidReference ) 
	<< "HepMC has null pointer to GenEvent" << endl;
    const size_t size = mc->particles_size();
    cout << "particles: " << size << endl;
  }
  InputTag src_;
};

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( DummyHepMCAnalyzer );


