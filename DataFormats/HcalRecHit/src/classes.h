#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalTriggerPrimitiveRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/SortedCollection.h"

namespace {
  namespace {
    std::vector<HBHERecHit> vHBHE_;
    std::vector<HORecHit> vHO_;
    std::vector<HFRecHit> vHF_;
    std::vector<ZDCRecHit> vZDC_;
    std::vector<HcalCalibRecHit> vcal_;
    std::vector<HcalTriggerPrimitiveRecHit> vHTP_;

    HBHERecHitCollection theHBHE_;
    HORecHitCollection theHO_;
    HFRecHitCollection theHF_;
    ZDCRecHitCollection theZDC_;
    HcalCalibRecHitCollection thecalib_;
    HcalTrigPrimRecHitCollection theHTP_;
    HcalSourcePositionData theSPD_;

    edm::Wrapper<HBHERecHitCollection> theHBHEw_;
    edm::Wrapper<HORecHitCollection> theHOw_;
    edm::Wrapper<HFRecHitCollection> theHFw_;
    edm::Wrapper<ZDCRecHitCollection> theZDCw_;
    edm::Wrapper<HcalCalibRecHitCollection> theCalibw_;
    edm::Wrapper<HcalTrigPrimRecHitCollection> theHTPw_;
    edm::Wrapper<HcalSourcePositionData> theSPDw_;

    edm::Ref<HBHERecHitCollection> theHBHEr_;
    edm::Ref<HORecHitCollection> theHOr_;
    edm::Ref<HFRecHitCollection> theHFr_;
    edm::Ref<ZDCRecHitCollection> theZDCr_;
    edm::Ref<HcalCalibRecHitCollection> theCalibr_;
    edm::Ref<HcalTrigPrimRecHitCollection> theHTPr_;

    edm::RefProd<HBHERecHitCollection> theHBHErp_;
    edm::RefProd<HORecHitCollection> theHOrp_;
    edm::RefProd<HFRecHitCollection> theHFrp_;
    edm::RefProd<ZDCRecHitCollection> theZDCrp_;
    edm::RefProd<HcalCalibRecHitCollection> theCalibrp_;
    edm::RefProd<HcalTrigPrimRecHitCollection> theHTPrp_;

    edm::RefVector<HBHERecHitCollection> theHBHErv_;
    edm::RefVector<HORecHitCollection> theHOrv_;
    edm::RefVector<HFRecHitCollection> theHFrv_;
    edm::RefVector<ZDCRecHitCollection> theZDCrv_;
    edm::RefVector<HcalCalibRecHitCollection> theCalibrv_;
    edm::RefVector<HcalTrigPrimRecHitCollection> theHTPrv_;
  }
}

