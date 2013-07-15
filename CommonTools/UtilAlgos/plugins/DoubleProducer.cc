/* \class DoubleProducer
 *
 * \author Luca Lista, INFN
 *
 * $Id: DoubleProducer.cc,v 1.1 2009/03/03 13:07:29 llista Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"

class DoubleProducer : public edm::EDProducer {
public:
  DoubleProducer( const edm::ParameterSet & cfg );
private:
  void produce( edm::Event & evt, const edm::EventSetup&) override;
  double value_;
};

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace std;

DoubleProducer::DoubleProducer( const ParameterSet & cfg ) :
value_( cfg.getParameter<double>( "value" ) ){
  produces<double>();
}

void DoubleProducer::produce( Event & evt, const EventSetup & ) {
  auto_ptr<double> value( new double( value_ ) );
  evt.put( value );
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( DoubleProducer );
