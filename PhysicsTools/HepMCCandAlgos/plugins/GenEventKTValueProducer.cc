/** \class GenEventKTValueProducer 
 *
 * \author Christophe Saout, CERN
 *
 * \version $Id: GenEventKTValueProducer.cc,v 1.2 2008/11/26 17:18:02 saout Exp $
 *
 */

#include <memory>
#include <vector>
#include <cmath>
#include <set>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

using namespace std;
using namespace edm;
using namespace reco;

class GenEventKTValueProducer : public EDProducer {
 public:
  /// constructor
  GenEventKTValueProducer( const ParameterSet & );

 private:
  void produce( Event& evt, const EventSetup& es );
  edm::InputTag src_;
};

namespace {
  struct MatrixElement {
    vector<const Candidate*> incoming;
    vector<const Candidate*> outgoing;
  };
}

// Funky shit to identify the actual incoming and outgoing partons in a
// Herwig++ matrix element documentation, which is actually represented
// as the Feynman graph with multiple vertices
// This code walks the whole graph and stops when it leaves status 3
// parts and also checks for general consistency and aborts if problems
// are detected
static bool findHerwigPPME( const Candidate *seed, MatrixElement &me )
{
  me.incoming.clear();
  me.outgoing.clear();

  set<const Candidate*> visited, seeds;

  visited.insert(seed);
  seeds.insert(seed);

  while( !seeds.empty() ) {
    set<const Candidate*> tmp;

    for( std::set<const Candidate*>::const_iterator iter = seeds.begin();
         iter != seeds.end(); ++iter ) {

      const Candidate *cand = *iter;
      unsigned int nMothers = cand->numberOfMothers();
      unsigned int nDaughters = cand->numberOfDaughters();

      // this cannot be the matrix element, since we reached open ends
      if ( !nMothers || !nDaughters )
        return false;

      if ( nMothers == 1 && cand->mother()->status() != 3 ) {
        // we found an incoming parton
        me.incoming.push_back( cand );
      } else {
        // check all mothers on the next iteration
        for( unsigned int i = 0; i < nMothers; i++ ) {
          const Candidate *mother = cand->mother( i );
          if ( mother->status() != 3 )
            continue;
          if ( visited.find( mother ) == visited.end() ) {
            visited.insert( mother );
            tmp.insert( mother );
          }
        }
      }

      if ( nDaughters == 1 && cand->daughter(0)->status() != 3 ) {
        // we found an outgoing parton
        me.outgoing.push_back( cand );
      } else {
        // check all daughters on the next iteration
        for( unsigned int i = 0; i < nDaughters; i++ ) {
          const Candidate *daughter = cand->daughter( i );
          if ( daughter->status() != 3 )
            continue;
          if ( visited.find( daughter ) == visited.end() ) {
            visited.insert( daughter );
            tmp.insert( daughter );
          }
        }
      }
    }

    swap(seeds, tmp);
  }

  return me.incoming.size() > 1 && me.outgoing.size() > 1;
}

static bool findHerwig6ME( const GenParticleCollection &c, MatrixElement &me )
{
  me.incoming.clear();
  me.outgoing.clear();

  for(GenParticleCollection::const_iterator p = c.begin(); p != c.end(); ++p)
    if (p->status() == 121 || p->status() == 122)
      me.incoming.push_back(&*p);
    else if (p->status() == 123 || p->status() == 124)
      me.outgoing.push_back(&*p);

  return me.incoming.size() > 1 && me.outgoing.size() > 1;
}

GenEventKTValueProducer::GenEventKTValueProducer( const ParameterSet & p ) :
  src_( p.getParameter<InputTag>( "src" ) )
{
  produces<double>();
}

void GenEventKTValueProducer::produce( Event& evt, const EventSetup& es )
{
  Handle<GenParticleCollection> genParticles;

  evt.getByLabel( src_, genParticles );

  auto_ptr<double> event_kt_value( new double( -1. ) );

  vector<const Candidate*> herwigPPMECandidates;

  // find the hard interaction(s)
  for(GenParticleCollection::const_iterator iter = genParticles->begin();
      iter != genParticles->end(); ++iter) {

    int status = iter->status();

    if ( status != 3 || iter->numberOfMothers() != 2 )
      continue;

    const Candidate *mothers[2] = { iter->mother(0), iter->mother(1) };

    // we only look at daughters once
    if (mothers[0]->daughter(0) != &*iter)
      continue;

    if ( mothers[0]->numberOfDaughters() > 1 ) {
      // we have a good "standard" documentation line
      double maxKT = -1.;
      for( Candidate::const_iterator iter2 = mothers[0]->begin();
           iter2 != mothers[0]->end(); ++iter2 ) {
        if ( iter2->status() == 3 &&
             ( iter2->mother(0) == mothers[1] ||
               iter2->mother(1) == mothers[1] ) )
          maxKT = max( maxKT, iter2->pt() );
      }

      if ( maxKT > 0. ) {
        *event_kt_value = maxKT;
        break;
      }
    } else {
      MatrixElement me;
      if ( !findHerwigPPME( &*iter, me ) )
        continue;

      // ok, we have found a Herwig++ matrix element with a subsection
      // of the status 2 graph with all one-mother-one-daughter endpoints...
      for( vector<const Candidate*>::const_iterator out = me.outgoing.begin();
           out != me.outgoing.end(); ++out )
        *event_kt_value = max( *event_kt_value, (*out)->pt() );
    }
  }

  if ( *event_kt_value < 0. ) {
    // last resort, it might be Herwig6
    MatrixElement me;
    if ( findHerwig6ME( *genParticles, me ) ) {
      for( vector<const Candidate*>::const_iterator out = me.outgoing.begin();
           out != me.outgoing.end(); ++out )
        *event_kt_value = max( *event_kt_value, (*out)->pt() );
    }
  }

  evt.put( event_kt_value );
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenEventKTValueProducer );
