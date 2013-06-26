#ifndef UtilAlgos_AndSelector_h
#define UtilAlgos_AndSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/AndSelector.h"

namespace reco {
  namespace modules {
    
    template<typename S1, typename S2, typename S3, typename S4, typename S5>
    struct ParameterAdapter<AndSelector<S1, S2, S3, S4, S5> > {
      static AndSelector<S1, S2, S3, S4, S5> make( const edm::ParameterSet & cfg ) {
	return AndSelector<S1, S2, S3, S4, S5>( modules::make<S1>( cfg ), 
						modules::make<S2>( cfg ), 
						modules::make<S3>( cfg ), 
						modules::make<S4>( cfg ), 
						modules::make<S5>( cfg ) ); 
      }
    };

    template<typename S1, typename S2, typename S3, typename S4>
    struct ParameterAdapter<AndSelector<S1, S2, S3, S4> > {
      static AndSelector<S1, S2, S3, S4> make( const edm::ParameterSet & cfg ) {
	return AndSelector<S1, S2, S3, S4>( modules::make<S1>( cfg ), 
					    modules::make<S2>( cfg ), 
					    modules::make<S3>( cfg ), 
					    modules::make<S4>( cfg ) ); 
      }
    };
    
    template<typename S1, typename S2, typename S3>
    struct ParameterAdapter<AndSelector<S1, S2, S3> > {
      static AndSelector<S1, S2, S3> make( const edm::ParameterSet & cfg ) {
	return AndSelector<S1, S2, S3>( modules::make<S1>( cfg ), 
					modules::make<S2>( cfg ), 
					modules::make<S3>( cfg ) ); 
      }
    };

    template<typename S1, typename S2>
    struct ParameterAdapter<AndSelector<S1, S2> > {
      static AndSelector<S1, S2> make( const edm::ParameterSet & cfg ) {
	return AndSelector<S1, S2>( modules::make<S1>( cfg ), 
				    modules::make<S2>( cfg ) ); 
      }
    };

  }
}

#endif

