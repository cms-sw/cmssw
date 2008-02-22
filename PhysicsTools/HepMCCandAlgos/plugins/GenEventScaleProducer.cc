/** \class GenEventScaleProducer 
 *
 * \author Luca Lista, INFN
 * \author Filip Moortgat, ETH
 *
 * \version $Id: GenEventScaleProducer.cc,v 1.1 2007/08/12 11:53:57 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include <vector>

namespace edm { class ParameterSet; }
namespace HepMC { class GenParticle; class GenEvent; }

class GenEventScaleProducer : public edm::EDProducer {
 public:
  /// constructor
  GenEventScaleProducer( const edm::ParameterSet & );

 private:
  void produce( edm::Event& evt, const edm::EventSetup& es );
  edm::InputTag src_;
};

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace edm;
using namespace std;
using namespace HepMC;

GenEventScaleProducer::GenEventScaleProducer( const ParameterSet & p ) :
  src_( p.getParameter<InputTag>( "src" ) ) {
  produces<double>();
}


void GenEventScaleProducer::produce( Event& evt, const EventSetup& es ) {
  Handle<HepMCProduct> mc;
  evt.getByLabel( src_, mc );
  const GenEvent * genEvt = mc->GetEvent();
  if( genEvt == 0 ) 
    throw edm::Exception( edm::errors::InvalidReference ) 
      << "HepMC has null pointer to GenEvent" << endl;
  auto_ptr<double> event_scale( new double(1) );
  (*event_scale) = genEvt->event_scale();
  evt.put( event_scale );
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenEventScaleProducer );

