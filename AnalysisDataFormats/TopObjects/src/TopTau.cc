//
// $Id: $
//

#include "AnalysisDataFormats/TopObjects/interface/TopTau.h"


/// default constructor
TopTau::TopTau() : TopLepton<TopTauType>(),emEnergyFraction_(0.),eOverP_(0.) {
}


/// constructor from TopTauType
TopTau::TopTau(const TopTauType & aTau):TopLepton<TopTauType>(aTau),emEnergyFraction_(0.),eOverP_(0.) {
}


/// destructor
TopTau::~TopTau() {
}

