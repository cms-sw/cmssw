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


namespace {
  namespace {
    std::vector<CaloTower> v1;
    CaloTowerCollection c1;
    CaloTowerPtr p1;
    CaloTowerRef r1;
    CaloTowersRef rr1;
    CaloTowerRefs rrr1;
    edm::Wrapper<CaloTowerCollection> w1;

    edm::reftobase::Holder<reco::Candidate, CaloTowerRef> rbh;
    edm::reftobase::RefHolder<CaloTowerRef> rbrh;
    edm::reftobase::VectorHolder<reco::Candidate, CaloTowerRefs> rbhs;
    edm::reftobase::RefVectorHolder<CaloTowerRefs> rbrhs;
   edm::Ptr<CaloTower> ptr;
   edm::PtrVector<CaloTower> ptrv;
  }
}
