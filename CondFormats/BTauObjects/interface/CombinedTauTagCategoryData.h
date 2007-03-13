#ifndef CombinedTauTagCategoryData_H
#define CombinedTauTagCategoryData_H

#include "DataFormats/BTauReco/interface/TaggingVariable.h"

struct CombinedTauTagCategoryData {
  int truthmatched1orfake0candidates,signaltks_n;
  float EtMin,EtMax;
  reco::TaggingVariable theTagVar;
};
#endif
