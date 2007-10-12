/** \class GenEventRunInfoProducer
 *
 * \author Liz,Oliver,Filip
 *
 * Save the cross section and filter efficiency info from the RUN section into the EVENT
 * Workaround for CSA07.
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include <vector>

namespace edm { class ParameterSet; }
namespace HepMC { class GenParticle; class GenEvent; }

class GenEventRunInfoProducer : public edm::EDProducer {
 public:
  /// constructor
  GenEventRunInfoProducer( const edm::ParameterSet & );

 private:
  void produce( edm::Event& evt, const edm::EventSetup& es );
  edm::InputTag src_;
};

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/HepMCProduct/interface/GenInfoProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace edm;
using namespace std;
using namespace HepMC;

GenEventRunInfoProducer::GenEventRunInfoProducer( const ParameterSet & p ) :
  src_( p.getParameter<InputTag>( "src" ) ) {
  produces<double>("AutoCrossSection");
  produces<double>("PreCalculatedCrossSection");
  produces<double>("FilterEfficiency");
}


void GenEventRunInfoProducer::produce( Event& evt, const EventSetup& es ) {

  // get the run info
  Handle<GenInfoProduct> gi;
  evt.getRun().getByLabel( src_, gi);
      
  auto_ptr<double> cs1( new double(1) );
  auto_ptr<double> cs2( new double(1) );
  auto_ptr<double> eff( new double(1) );
   
  double cross_section1 = gi->cross_section(); // automatically calculated at the end of the Pythia/Herwig run 
  double cross_section2 = gi->external_cross_section(); // is the precalculated one written in the cfg file -- units is pb!!
  double filter_eff = gi->filter_efficiency();
  
  (*cs1) = cross_section1;
  (*cs2) = cross_section2;
  (*eff) = filter_eff;
  evt.put( cs1, "AutoCrossSection" );
  evt.put( cs2, "PreCalculatedCrossSection" );
  evt.put( eff, "FilterEfficiency" );
  
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenEventRunInfoProducer );
