#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <ostream>
#include <sstream>
#include <algorithm>

/*----------------------------------------------------------------------

$Id: EventAuxiliary.cc,v 1.1 2007/03/04 04:48:09 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  void
  EventAuxiliary::write(std::ostream& os) const {
    os << "Process History ID = " <<  processHistoryID_ << std::endl;
    os << id_ << std::endl;
    //os << "TimeStamp = " << time_ << std::endl;
    os << "LuminosityBlockNumber_t = " << luminosityBlock_ << std::endl;
  }

  std::string &
  EventAuxiliary::checkExperimentType( const std::string & eType ) {
    static svec IDstrings;
    static bool firstCall = true;

// On the first call, load up the vector of allowed strings.
    if(firstCall) {
      IDstrings.push_back(std::string("DAQ"));
      IDstrings.push_back(std::string("Testing"));
      IDstrings.push_back(std::string("Unspecified"));
      IDstrings.push_back(std::string("Cosmics"));
      IDstrings.push_back(std::string("ParticleGun"));
      IDstrings.push_back(std::string("Geant4"));
      IDstrings.push_back(std::string("Pythia"));
      firstCall = false;
    }

    svec::iterator it;
    it = find(IDstrings.begin(), IDstrings.end(), eType);
    if( it != IDstrings.end() ) return *it;

// Throw an edm::Exception if the experimentType is not in our "approved" list.
   std::ostringstream os;
   os << "Fatal EventAuxiliary error  -- experimentType " << eType << " not found ";
   edm::Exception except(edm::errors::NotFound, os.str());
   throw except;
  }
}
