/* \class MassKinFitterCandProducer
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "PhysicsTools/RecoUtils/interface/CandMassKinFitter.h"
#include <vector>

class MassKinFitterCandProducer : public edm::EDProducer {
public:
  explicit MassKinFitterCandProducer( const edm::ParameterSet & );

private:
  edm::InputTag src_;
  CandMassKinFitter fitter_;
  void produce( edm::Event &, const edm::EventSetup & );
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Candidate/interface/Candidate.h"

MassKinFitterCandProducer::MassKinFitterCandProducer( const edm::ParameterSet & cfg ) :
  src_( cfg.getParameter<edm::InputTag>( "src" ) ),
  fitter_( cfg.getParameter<double>( "mass" ) ) {
  produces<reco::CandidateCollection>();
}

void MassKinFitterCandProducer::produce( edm::Event & evt, const edm::EventSetup & es ) {
  using namespace edm; 
  using namespace reco;
  Handle<CandidateCollection> cands;
  evt.getByLabel( src_, cands );
  std::auto_ptr<CandidateCollection> refitted( new CandidateCollection );
  for( CandidateCollection::const_iterator c = cands->begin(); c != cands->end(); ++ c ) {
    Candidate * clone = c->clone();
    fitter_.set( * clone );
    refitted->push_back( clone );
  }
  evt.put( refitted );
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MassKinFitterCandProducer );

