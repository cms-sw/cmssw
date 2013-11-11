#ifndef IsolationAlgos_IsolationProducer_h
#define IsolationAlgos_IsolationProducer_h
/* \class IsolationProducer<C1, C2, Algo>
 *
 * \author Francesco Fabozzi, INFN
 *
 * template class to store isolation
 *
 */
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include <vector>

namespace helper {

  template<typename Alg>
  struct NullIsolationAlgorithmSetup {
    static void init( Alg &, const edm::EventSetup& ) { }
  };

  template<typename Alg>
  struct IsolationAlgorithmSetup {
    typedef NullIsolationAlgorithmSetup<Alg> type;
  };
}

template <typename C1, typename C2, typename Alg, 
	  typename OutputCollection = edm::AssociationVector<edm::RefProd<C1>, 
							     std::vector<typename Alg::value_type> >,
 	  typename Setup = typename helper::IsolationAlgorithmSetup<Alg>::type>
class IsolationProducer : public edm::EDProducer {
public:
  IsolationProducer( const edm::ParameterSet & );
  ~IsolationProducer();

private:
  void produce( edm::Event&, const edm::EventSetup& );
  edm::InputTag src_, elements_;
  Alg alg_;
};

template <typename C1, typename C2, typename Alg, typename OutputCollection, typename Setup>
IsolationProducer<C1, C2, Alg, OutputCollection, Setup>::IsolationProducer( const edm::ParameterSet & cfg ) :
  src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
  elements_( cfg.template getParameter<edm::InputTag>( "elements" ) ),
  alg_( reco::modules::make<Alg>( cfg ) ) {
  produces<OutputCollection>();
}

template <typename C1, typename C2, typename Alg, typename OutputCollection, typename Setup>
IsolationProducer<C1, C2, Alg, OutputCollection, Setup>::~IsolationProducer() {
}

template <typename C1, typename C2, typename Alg, typename OutputCollection, typename Setup>
void IsolationProducer<C1, C2, Alg, OutputCollection, Setup>::produce( edm::Event& evt, const edm::EventSetup& es ) {
  using namespace edm;
  using namespace std;
  Handle<C1> src;
  Handle<C2> elements;
  evt.getByLabel( src_, src );
  evt.getByLabel( elements_, elements );

  Setup::init( alg_, es );

  typename OutputCollection::refprod_type ref( src );
  auto_ptr<OutputCollection> isolations( new OutputCollection( ref )  );

  size_t i = 0;
  for( typename C1::const_iterator lep = src->begin(); lep != src->end(); ++ lep ) {
    typename Alg::value_type iso= alg_(*lep,*elements); 
    isolations->setValue( i++, iso );
  }
  evt.put( isolations );
}

#endif
