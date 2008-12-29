#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>
using namespace std;
using namespace edm;
using namespace HepMC;

class DummyHepMCAnalyzer : public EDAnalyzer {
private: 
  bool dumpHepMC_;
public:
  explicit DummyHepMCAnalyzer( const ParameterSet & cfg ) : 
    dumpHepMC_( cfg.getUntrackedParameter<bool>( "dumpHepMC", false ) ),
    src_( cfg.getParameter<InputTag>( "src" ) )
  {
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
    if ( dumpHepMC_ ) mc->print( std::cout );
  }
  InputTag src_;
};

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( DummyHepMCAnalyzer );


