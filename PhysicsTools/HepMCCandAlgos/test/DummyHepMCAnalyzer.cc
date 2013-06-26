#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>
using namespace std;
using namespace edm;
using namespace HepMC;

class DummyHepMCAnalyzer : public EDAnalyzer {
private: 
  bool dumpHepMC_;
  bool dumpPDF_;
  bool checkPDG_;
public:
  explicit DummyHepMCAnalyzer( const ParameterSet & cfg ) : 
    dumpHepMC_( cfg.getUntrackedParameter<bool>( "dumpHepMC", false ) ),
    dumpPDF_( cfg.getUntrackedParameter<bool>( "dumpPDF", false ) ),
    checkPDG_( cfg.getUntrackedParameter<bool>( "checkPDG", false ) ),
    src_( cfg.getParameter<InputTag>( "src" ) )
  {
  }
private:
  void analyze( const Event & evt, const EventSetup& es ) override {
    Handle<HepMCProduct> hepMC;
    evt.getByLabel( src_, hepMC );
    const GenEvent * mc = hepMC->GetEvent();
    if( mc == 0 ) 
      throw edm::Exception( edm::errors::InvalidReference ) 
	<< "HepMC has null pointer to GenEvent" << endl;
    const size_t size = mc->particles_size();
    cout << "\n particles #: " << size << endl;
    if ( dumpPDF_ ) std::cout << "\n PDF info: " << mc->pdf_info() << std::endl;
    if ( dumpHepMC_ ) mc->print( std::cout );
    if ( checkPDG_ ) {
      edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
      es.getData( fPDGTable );
      for ( HepMC::GenEvent::particle_const_iterator part = mc->particles_begin();
            part != mc->particles_end(); ++part ) 
        {
          const HepPDT::ParticleData* 
            PData = fPDGTable->particle(HepPDT::ParticleID((*part)->pdg_id())) ;
          if ( !PData ) std::cout << "Missing entry in particle table for PDG code = " << (*part)->pdg_id() << std::endl;
        }
    }
  }
  InputTag src_;
};

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( DummyHepMCAnalyzer );



