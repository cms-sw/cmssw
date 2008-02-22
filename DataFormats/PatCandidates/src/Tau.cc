//
// $Id: Tau.cc,v 1.2 2008/01/22 21:58:16 lowette Exp $
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
Tau::Tau(const edm::RefToBase<TauType> & aTauRef) : Lepton<TauType>(aTauRef) {
}


/// destructor
Tau::~Tau() {
}
