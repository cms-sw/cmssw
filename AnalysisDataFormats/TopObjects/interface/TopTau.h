//
// $Id$
//

#ifndef TopObjects_TopTau_h
#define TopObjects_TopTau_h


#include "DataFormats/TauReco/interface/Tau.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"


//typedef reco::IsolatedTauTagInfo TopTauType;
typedef reco::Tau TopTauType;


/// definition of TopTau as a TopLepton of TopTauType
typedef TopLepton<TopTauType> TopTau;


#endif
