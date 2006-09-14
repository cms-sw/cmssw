#ifndef MuonReco_MuIsoDepositFwd_h
#define MuonReco_MuIsoDepositFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {
  class MuIsoDeposit;
  /// collection of MuIsoDeposit objects
  typedef std::vector<MuIsoDeposit> MuIsoDepositCollection;
  /// presistent reference to a MuIsoDeposit
  typedef edm::Ref<MuIsoDepositCollection> MuIsoDepositRef;
  /// references to MuIsoDeposit collection
  typedef edm::RefProd<MuIsoDepositCollection> MuIsoDepositRefProd;
  /// vector of references to MuIsoDeposit objects all in the same collection
  typedef edm::RefVector<MuIsoDepositCollection> MuIsoDepositRefVector;
  /// iterator over a vector of references to MuIsoDeposit objects all in the same collection
  typedef MuIsoDepositRefVector::iterator MuIsoDeposit_iterator;

  /// Map
  class MuIsoSimpleDeposit;
  typedef edm::AssociationMap<edm::OneToValue<TrackCollection,MuIsoSimpleDeposit> > MuIsoAssociationMap;
}

#endif
