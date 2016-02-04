#include "DataFormats/Candidate/interface/const_iterator.h"
#include "DataFormats/Candidate/interface/iterator.h"

using namespace reco::candidate;

const_iterator::const_iterator( const iterator & it ) : 
  i( it.i->const_clone() ) { 
}
