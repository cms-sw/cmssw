/** \class GenEventWeightProducer 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: GenEventWeightProducer.cc,v 1.1 2007/06/12 11:53:57 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include <vector>

namespace edm { class ParameterSet; }
namespace HepMC { class GenParticle; class GenEvent; }

class GenEventWeightProducer : public edm::EDProducer {
 public:
  /// constructor
  GenEventWeightProducer( const edm::ParameterSet & );

 private:
  void produce( edm::Event& evt, const edm::EventSetup& es );
  edm::InputTag src_;
};

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace edm;
using namespace std;
using namespace HepMC;

GenEventWeightProducer::GenEventWeightProducer( const ParameterSet & p ) :
  src_( p.getParameter<InputTag>( "src" ) ) {
  produces<double>();
}


void GenEventWeightProducer::produce( Event& evt, const EventSetup& es ) {
  Handle<HepMCProduct> mc;
  evt.getByLabel( src_, mc );
  const GenEvent * genEvt = mc->GetEvent();
  if( genEvt == 0 ) 
    throw edm::Exception( edm::errors::InvalidReference ) 
      << "HepMC has null pointer to GenEvent" << endl;
  auto_ptr<double> weight( new double(1) );
  HepMC::WeightContainer wc = genEvt->weights();
  if ( wc.size() > 0 )  (*weight) = wc[ 0 ];
  evt.put( weight );
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenEventWeightProducer );

