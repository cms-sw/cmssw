#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"
#include "DataFormats/Common/interface/VectorHolder.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/AtomicPtrCache.h"


namespace DataFormats_CaloTowers {
  struct dictionary {
    std::vector<CaloTower> v1;
    CaloTowerCollection c1;
    CaloTowerPtr p1;
    CaloTowerFwdPtr fp1;
    CaloTowerRef r1;
    CaloTowersRef rr1;
    CaloTowerRefs rrr1;
    CaloTowerFwdRef rrrr1;
    CaloTowerFwdRefVector rrrrv1;
    CaloTowerFwdPtr rrrrr1;
    CaloTowerFwdPtrVector rrrrrv1;
    edm::Wrapper<CaloTowerCollection> w1;
    edm::Wrapper<CaloTowerFwdRefVector> w2;
    edm::Wrapper<CaloTowerFwdPtrVector> w3;
    edm::Wrapper< std::vector<CaloTower> > w4;

    edm::reftobase::Holder<reco::Candidate, CaloTowerRef> rbh;
    edm::reftobase::RefHolder<CaloTowerRef> rbrh;
    edm::reftobase::VectorHolder<reco::Candidate, CaloTowerRefs> rbhs;
    edm::reftobase::RefVectorHolder<CaloTowerRefs> rbrhs;

    edm::PtrVector<CaloTower> ptrv;
    std::vector<edm::Ptr<CaloTower> > vp1;

    std::vector<CaloTowerDetId> vctdi;
    edm::AtomicPtrCache<std::vector<CaloTowerPtr> > easvrp;
  };
}
