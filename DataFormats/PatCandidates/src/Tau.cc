//
// $Id: Tau.cc,v 1.1 2008/01/15 12:59:35 lowette Exp $
//

#include "DataFormats/PatCandidates/interface/Tau.h"


using namespace pat;


/// default constructor
Tau::Tau() : Lepton<TauType>(), emEnergyFraction_(0.), eOverP_(0.) {
}


/// constructor from TauType
Tau::Tau(const TauType & aTau) : Lepton<TauType>(aTau), emEnergyFraction_(0.), eOverP_(0.) {
}


/// constructor from ref to TauType
Tau::Tau(const edm::Ref<std::vector<TauType> > & aTauRef) : Lepton<TauType>(aTauRef) {
}


/// destructor
Tau::~Tau() {
}
