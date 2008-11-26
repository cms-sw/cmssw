/** \class GenEventKTValueProducer 
 *
 * \author Christophe Saout, CERN
 *
 * \version $Id: GenEventKTValueProducer.cc,blubb Exp $
 *
 */

#include <memory>
#include <vector>
#include <cmath>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

class GenEventKTValueProducer : public edm::EDProducer {
 public:
  /// constructor
  GenEventKTValueProducer( const edm::ParameterSet & );

 private:
  void produce( edm::Event& evt, const edm::EventSetup& es );
  edm::InputTag src_;
};

using namespace std;
using namespace edm;
using namespace reco;

GenEventKTValueProducer::GenEventKTValueProducer( const ParameterSet & p ) :
  src_( p.getParameter<InputTag>( "src" ) )
{
  produces<double>();
}


static bool allowedParton( int pdgId )
{
  return (std::abs(pdgId) % 1000000) < 80;
}

void GenEventKTValueProducer::produce( Event& evt, const EventSetup& es )
{
  Handle<GenParticleCollection> genParticles;

  evt.getByLabel( src_, genParticles );

  double bestMass2 = -1.;
  const Candidate *bestMothers[2] = { 0, 0 };

  // find the hard interaction
  for(GenParticleCollection::const_iterator iter = genParticles->begin();
      iter != genParticles->end(); ++iter) {
    if ( iter->status() == 1 ||
         !allowedParton( iter->pdgId() ) ||
         iter->numberOfMothers() != 2 )
      continue;

    const Candidate *mothers[2] = { iter->mother(0), iter->mother(1) };
    double mass2 = ( mothers[0]->p4() + mothers[1]->p4() ).M2();
    if (mass2 > bestMass2) {
      bestMass2 = mass2;
      bestMothers[0] = mothers[0];
      bestMothers[1] = mothers[1];
    }
  }

  auto_ptr<double> event_kt_value( new double( -1. ) );

  if ( bestMothers[0] ) {
    // now loop over all daughters that have those two mothers
    for(Candidate::const_iterator iter = bestMothers[0]->begin();
        iter != bestMothers[0]->end(); ++iter) {
      if ( iter->numberOfMothers() == 2 &&
           ( iter->mother(1) == bestMothers[1] ||
             iter->mother(0) == bestMothers[1] ) &&
           allowedParton( iter->pdgId() ) )
        *event_kt_value = std::max( *event_kt_value, iter->pt() );
    }
  }

  std::cout << "kt_value = " << *event_kt_value << std::endl;
  evt.put( event_kt_value );
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenEventKTValueProducer );
