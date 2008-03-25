#ifndef MuonReco_MuIsoDepositFwd_h
#define MuonReco_MuIsoDepositFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/ValueMap.h"


#include "DataFormats/Candidate/interface/CandidateFwd.h"

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

  /// Maps
  typedef edm::AssociationMap<edm::OneToValue<TrackCollection,bool> > MuIsoAssociationMap;
  typedef edm::AssociationMap<edm::OneToValue<TrackCollection,int> > MuIsoIntAssociationMap;
  typedef edm::AssociationMap<edm::OneToValue<TrackCollection,float> > MuIsoFloatAssociationMap;
  typedef edm::AssociationMap<edm::OneToValue<TrackCollection,reco::MuIsoDeposit> > MuIsoDepositAssociationMap;
  
  typedef edm::AssociationVector<CandidateBaseRefProd,MuIsoDepositCollection > MuIsoDepositAssociationVectorToCandidateView;
  typedef edm::AssociationVector<CandidateBaseRefProd,MuIsoDepositCollection > CandIsoDepositAssociationVector;
  typedef CandIsoDepositAssociationVector::value_type CandIsoDepositAssociationPair;

  typedef edm::ValueMap<reco::MuIsoDeposit> IsoDepositMap;

}

#endif
