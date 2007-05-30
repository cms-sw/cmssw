#ifndef IsolationAlgos_IsolationProducer_h
#define IsolationAlgos_IsolationProducer_h
/* \class IsolationProducer<C1, C2, Algo>
 *
 * \author Francesco Fabozzi, INFN
 *
 * template class to store isolation
 *
 */
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
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
	  typename Setup = typename helper::IsolationAlgorithmSetup<Alg>::type>
class IsolationProducer : public edm::EDProducer {
public:
  IsolationProducer( const edm::ParameterSet & );
  ~IsolationProducer();

private:
  void produce( edm::Event&, const edm::EventSetup& );
  edm::InputTag src_, elements_;
  Alg alg_;
  typedef typename Alg::value_type value_type;
  typedef edm::AssociationVector<edm::RefProd<C1>, 
				 std::vector<value_type> > IsolationCollection;
};

template <typename C1, typename C2, typename Alg, typename Setup>
IsolationProducer<C1, C2, Alg, Setup>::IsolationProducer( const edm::ParameterSet & cfg ) :
  src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
  elements_( cfg.template getParameter<edm::InputTag>( "elements" ) ),
  alg_( reco::modules::make<Alg>( cfg ) ) {
  //produces<IsolationCollection>();
  produces<std::vector<value_type> >();  //CMSSW_1_3_x
}

template <typename C1, typename C2, typename Alg, typename Setup>
IsolationProducer<C1, C2, Alg, Setup>::~IsolationProducer() {
}

template <typename C1, typename C2, typename Alg, typename Setup>
void IsolationProducer<C1, C2, Alg, Setup>::produce( edm::Event& evt, const edm::EventSetup& es ) {
  using namespace edm;
  using namespace std;
  Handle<C1> src;
  Handle<C2> elements;
  evt.getByLabel( src_, src );
  evt.getByLabel( elements_, elements );

  Setup::init( alg_, es );

  //auto_ptr<IsolationCollection> isolations( new IsolationCollection( edm::RefProd<C1>( src ) )  );
  auto_ptr<std::vector<value_type> > isolations( new std::vector<value_type>); //CMSSW_1_3_x

  //size_t i = 0;
  for( typename C1::const_iterator lep = src->begin(); lep != src->end(); ++ lep ) {
    value_type iso= alg_(*lep,*elements); 
    //isolations->setValue( i++, iso );
    isolations->push_back( iso );       //CMSSW_1_3_x
  }
  evt.put( isolations );
}

#endif
