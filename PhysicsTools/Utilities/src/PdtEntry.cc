#include "PhysicsTools/Utilities/interface/PdtEntry.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace std;

int PdtEntry::pdgId() const {
  if ( pdgId_ == 0 )
    throw cms::Exception( "ConfigError" )
      << "PdtEntry::pdgId was not set.";
  return pdgId_;
}

const string & PdtEntry::name() const {
  if ( name_.empty() )
    throw cms::Exception( "ConfigError" )
      << "PdtEntry::name was not set.";
  return name_;
}

