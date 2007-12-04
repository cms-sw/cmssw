#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "Math/PxPyPzE4D.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h" 
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h" 
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/RecoCandidate/interface/FitResult.h"
#include "DataFormats/RecoCandidate/interface/CaloRecHitCandidate.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/RecoCandidate/interface/TrackCandidateAssociation.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h" 
#include "DataFormats/HcalRecHit/interface/HcalRecHitFwd.h" 
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToValue.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"

namespace {
  namespace {
    reco::RecoChargedCandidateCollection v1;
    edm::Wrapper<reco::RecoChargedCandidateCollection> w1;
    edm::Ref<reco::RecoChargedCandidateCollection> r1;
    edm::RefProd<reco::RecoChargedCandidateCollection> rp1;
    edm::RefVector<reco::RecoChargedCandidateCollection> rv1;

    reco::RecoEcalCandidateCollection v2;
    edm::Wrapper<reco::RecoEcalCandidateCollection> w2;
    edm::Ref<reco::RecoEcalCandidateCollection> r2;
    edm::RefProd<reco::RecoEcalCandidateCollection> rp2;
    edm::RefVector<reco::RecoEcalCandidateCollection> rv2;

    reco::RecoEcalCandidateIsolationMap v3;
    edm::Wrapper<reco::RecoEcalCandidateIsolationMap> w3;
    edm::helpers::Key<edm::RefProd<reco::RecoEcalCandidateCollection > > h3;


    edm::reftobase::Holder<reco::Candidate, reco::RecoEcalCandidateRef> rb1;
    edm::reftobase::Holder<reco::Candidate, reco::RecoChargedCandidateRef> rb2;
    edm::reftobase::Holder<CaloRecHit, HBHERecHitRef> rb4;
    edm::reftobase::Holder<CaloRecHit, HORecHitRef > rb5;
    edm::reftobase::Holder<CaloRecHit, HFRecHitRef> rb6;
    edm::reftobase::Holder<CaloRecHit, ZDCRecHitRef> rb7;
    edm::reftobase::Holder<CaloRecHit, EcalRecHitRef> rb8;
    edm::RefToBase<CaloRecHit> rbh3;

    edm::Wrapper<reco::FitResultCollection> wfr1;
    edm::Wrapper<reco::TrackCandidateAssociation> tca1;

    edm::Wrapper<reco::SimToRecoCollection > ii2;
    reco::SimToRecoAssociation ii3;
    reco::SimToRecoAssociationRef ii4;
    reco::SimToRecoAssociationRefProd ii5;
    reco::SimToRecoAssociationRefVector ii6;

    edm::Wrapper<reco::RecoToSimCollection > jj2;
    reco::RecoToSimAssociation jj3;
    reco::RecoToSimAssociationRef jj4;
    reco::RecoToSimAssociationRefProd jj5;
    reco::RecoToSimAssociationRefVector jj6;
          
    edm::reftobase::Holder<reco::Candidate, reco::RecoChargedCandidateRef> rbc1;
    edm::reftobase::RefHolder<reco::RecoChargedCandidateRef> rbc2;
    edm::reftobase::VectorHolder<reco::Candidate, reco::RecoChargedCandidateRefVector> rbc3;
    edm::reftobase::RefVectorHolder<reco::RecoChargedCandidateRefVector> rbc4;
          
    edm::reftobase::Holder<reco::Candidate, reco::RecoEcalCandidateRef> rbe1;
    edm::reftobase::RefHolder<reco::RecoEcalCandidateRef> rbe2;
    edm::reftobase::VectorHolder<reco::Candidate, reco::RecoEcalCandidateRefVector> rbe3;
    edm::reftobase::RefVectorHolder<reco::RecoEcalCandidateRefVector> rbe4;
  }
}
