#ifndef UtilAlgos_NullPostProcessor_h
#define UtilAlgos_NullPostProcessor_h
/* \class helper::NullPostProcessor<OutputCollection>
 *
 * \author Luca Lista, INFN
 */
#include "FWCore/Framework/interface/ConsumesCollector.h"
namespace helper {

  template<typename OutputCollection>
  struct NullPostProcessor {
    NullPostProcessor( const edm::ParameterSet & iConfig, edm::ConsumesCollector && iC ) :
      NullPostProcessor( iConfig ) { }
    NullPostProcessor( const edm::ParameterSet & iConfig ) { }
    void init( edm::EDFilter & ) { }
    void process( edm::OrphanHandle<OutputCollection>, edm::Event & ) { }
  };

}

#endif

