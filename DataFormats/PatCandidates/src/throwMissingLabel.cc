#include "DataFormats/PatCandidates/interface/throwMissingLabel.h"

#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

namespace pat {
  void throwMissingLabel(const std::string& what, const std::string& bad_label,
                         const std::vector<std::string>& available) {
    cms::Exception ex(std::string("Unknown")+what);
    ex << "Requested " << what << " " << bad_label 
       << " is not available! Possible " << what << "s are: " << std::endl;
    for( const auto& name : available ) {
      ex << name << ' ';
    }
    throw ex;
  }
}
