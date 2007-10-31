#ifndef RecoAlgos_ConstrainedFitCandProducer_h
#define RecoAlgos_ConstrainedFitCandProducer_h
/* \class ConstrainedFitProducer
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/RecoCandidate/interface/FitResult.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include <vector>

template<typename Fitter,
	 typename Init = typename ::reco::modules::EventSetupInit<Fitter>::type>
class ConstrainedFitCandProducer : public edm::EDProducer {
public:
  explicit ConstrainedFitCandProducer( const edm::ParameterSet & );

private:
  edm::InputTag src_;
  bool saveFitResults_;
  Fitter fitter_;
  void produce( edm::Event &, const edm::EventSetup & );
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/EventSetup.h"

template<typename Fitter, typename Init>
ConstrainedFitCandProducer<Fitter, Init>::ConstrainedFitCandProducer( const edm::ParameterSet & cfg ) :
  src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
  saveFitResults_( cfg.template getParameter<bool>( "saveFitResults" ) ),
  fitter_( reco::modules::make<Fitter>( cfg ) ) {
  produces<reco::CandidateCollection>();
  std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
  if ( saveFitResults_ )
    produces<reco::FitResultCollection>().setBranchAlias( alias + "FitResults" );
}

template<typename Fitter, typename Init>
void ConstrainedFitCandProducer<Fitter, Init>::produce( edm::Event & evt, const edm::EventSetup & es ) {
  using namespace edm; 
  using namespace reco;
  Init::init( fitter_, es );
  Handle<CandidateCollection> cands;
  evt.getByLabel( src_, cands );
  FitQuality fq;
  std::auto_ptr<CandidateCollection> refitted( new CandidateCollection );
  std::auto_ptr<FitResultCollection> fitResults(new FitResultCollection(RefProd<CandidateCollection>(cands)) );
  size_t i = 0;
  for( CandidateCollection::const_iterator c = cands->begin(); c != cands->end(); ++ c ) {
    Candidate * clone = c->clone();
    fq = fitter_.set( * clone );
    refitted->push_back( clone );
    if ( saveFitResults_ )
      fitResults->setValue( i, fq );
  }
  evt.put( refitted );
  if (  saveFitResults_ )
    evt.put( fitResults );
}

#endif
