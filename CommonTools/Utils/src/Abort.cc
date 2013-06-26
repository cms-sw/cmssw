#include "CommonTools/Utils/src/Abort.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace reco::parser;

void Abort::operator()( const char *, const char * ) const {
  throw edm::Exception( edm::errors::Configuration,
			std::string( "parse rule not implemented yet" ) );
}
