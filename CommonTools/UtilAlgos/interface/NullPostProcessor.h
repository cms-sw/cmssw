#ifndef UtilAlgos_NullPostProcessor_h
#define UtilAlgos_NullPostProcessor_h
/* \class helper::NullPostProcessor<OutputCollection>
 *
 * \author Luca Lista, INFN
 */
namespace helper {

  template<typename OutputCollection>
  struct NullPostProcessor {
    NullPostProcessor( const edm::ParameterSet & ) { }
    void init( edm::EDFilter & ) { }
    void process( edm::OrphanHandle<OutputCollection>, edm::Event & ) { }
  };

}

#endif

