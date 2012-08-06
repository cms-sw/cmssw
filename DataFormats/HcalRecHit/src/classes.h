#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include <boost/cstdint.hpp> 
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h" 
#include "DataFormats/HcalRecHit/interface/HcalRecHitFwd.h" 
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h" 
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace {
  struct dictionary {
    std::vector<HBHERecHit> vHBHE_;
    std::vector<HORecHit> vHO_;
    std::vector<HFRecHit> vHF_;
    std::vector<ZDCRecHit> vZDC_;
    std::vector<CastorRecHit> vCastor_;
    std::vector<HcalCalibRecHit> vcal_;
    std::vector<HcalUpgradeRecHit> vUpgrade_;

    HBHERecHitCollection theHBHE_;
    HORecHitCollection theHO_;
    HFRecHitCollection theHF_;
    ZDCRecHitCollection theZDC_;
    CastorRecHitCollection theCastor_;
    HcalCalibRecHitCollection thecalib_;
    HcalUpgradeRecHitCollection theupgrade_;
    HcalSourcePositionData theSPD_;

    edm::Wrapper<HBHERecHitCollection> theHBHEw_;
    edm::Wrapper<HORecHitCollection> theHOw_;
    edm::Wrapper<HFRecHitCollection> theHFw_;
    edm::Wrapper<ZDCRecHitCollection> theZDCw_;
    edm::Wrapper<CastorRecHitCollection> theCastorw_;
    edm::Wrapper<HcalCalibRecHitCollection> theCalibw_;
    edm::Wrapper<HcalUpgradeRecHitCollection> theUpgradew_;
    edm::Wrapper<HcalSourcePositionData> theSPDw_;

    edm::Ref<HBHERecHitCollection> theHBHEr_;
    edm::Ref<HORecHitCollection> theHOr_;
    edm::Ref<HFRecHitCollection> theHFr_;
    edm::Ref<ZDCRecHitCollection> theZDCr_;
    edm::Ref<CastorRecHitCollection> theCastorr_;
    edm::Ref<HcalCalibRecHitCollection> theCalibr_;
    edm::Ref<HcalUpgradeRecHitCollection> theUpgrader_;

    edm::RefProd<HBHERecHitCollection> theHBHErp_;
    edm::RefProd<HORecHitCollection> theHOrp_;
    edm::RefProd<HFRecHitCollection> theHFrp_;
    edm::RefProd<ZDCRecHitCollection> theZDCrp_;
    edm::RefProd<CastorRecHitCollection> theCastorrp_;
    edm::RefProd<HcalCalibRecHitCollection> theCalibrp_;
    edm::RefProd<HcalUpgradeRecHitCollection> theUpgraderp_;

    edm::RefVector<HBHERecHitCollection> theHBHErv_;
    edm::RefVector<HORecHitCollection> theHOrv_;
    edm::RefVector<HFRecHitCollection> theHFrv_;
    edm::RefVector<ZDCRecHitCollection> theZDCrv_;
    edm::RefVector<CastorRecHitCollection> theCastorrv_;
    edm::RefVector<HcalCalibRecHitCollection> theCalibrv_;
    edm::RefVector<HcalUpgradeRecHitCollection> theUpgraderv_;

    edm::reftobase::Holder<CaloRecHit, HBHERecHitRef> rb4;
    edm::reftobase::Holder<CaloRecHit, HORecHitRef > rb5;
    edm::reftobase::Holder<CaloRecHit, HFRecHitRef> rb6;
    edm::reftobase::Holder<CaloRecHit, ZDCRecHitRef> rb7;
    edm::reftobase::Holder<CaloRecHit, HcalUpgradeRecHitRef> rb8;
  };
}

