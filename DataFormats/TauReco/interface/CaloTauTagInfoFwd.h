#ifndef DataFormats_TauReco_CaloTauTagInfoFwd_h
#define DataFormats_TauReco_CaloTauTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class CaloTauTagInfo;
  /// collection of CaloTauTagInfo objects
  typedef std::vector<CaloTauTagInfo> CaloTauTagInfoCollection;
  /// presistent reference to a CaloTauTagInfo
  typedef edm::Ref<CaloTauTagInfoCollection> CaloTauTagInfoRef;
  /// references to CaloTauTagInfo collection
  typedef edm::RefProd<CaloTauTagInfoCollection> CaloTauTagInfoRefProd;
  /// vector of references to CaloTauTagInfo objects all in the same collection
  typedef edm::RefVector<CaloTauTagInfoCollection> CaloTauTagInfoRefVector;
  /// iterator over a vector of references to CaloTauTagInfo objects all in the same collection
  typedef CaloTauTagInfoRefVector::iterator calotautaginfo_iterator;
}

#endif
