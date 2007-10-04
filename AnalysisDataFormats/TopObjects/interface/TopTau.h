//
// $Id: TopTau.h,v 1.1 2007/09/20 18:12:22 lowette Exp $
//

#ifndef TopObjects_TopTau_h
#define TopObjects_TopTau_h


#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"


//typedef reco::IsolatedTauTagInfo TopTauType;
typedef reco::BaseTau TopTauType;


/// definition of TopTau as a TopLepton of TopTauType
typedef TopLepton<TopTauType> TopTau;


#endif
