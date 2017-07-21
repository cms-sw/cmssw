
#include <vector>
#include <boost/cstdint.hpp> 
#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/L1TriggerRates.h"
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "DataFormats/Scalers/interface/TimeSpec.h"
#include "DataFormats/Scalers/interface/BSTRecord.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace DataFormats_Scalers {
  struct dictionary {
    L1AcceptBunchCrossing l1AcceptBunchCrossing;
    L1TriggerScalers l1TriggerScalers;
    L1TriggerRates l1TriggerRates;
    Level1TriggerScalers level1TriggerScalers;
    Level1TriggerRates level1TriggerRates;
    LumiScalers lumiScalers;
    DcsStatus dcsStatus;
    BeamSpotOnline beamSpotOnline;
    BSTRecord bstRecord;

    edm::Wrapper<L1AcceptBunchCrossing> w_l1AcceptBunchCrossing;
    edm::Wrapper<L1TriggerScalers> w_l1TriggerScalers;
    edm::Wrapper<L1TriggerRates> w_l1TriggerRates;
    edm::Wrapper<Level1TriggerScalers> w_level1TriggerScalers;
    edm::Wrapper<Level1TriggerRates> w_level1TriggerRates;
    edm::Wrapper<LumiScalers> w_lumiScalers;
    edm::Wrapper<DcsStatus> w_dcsStatus;
    edm::Wrapper<BeamSpotOnline> w_beamSpotOnline;
    edm::Wrapper<BSTRecord> w_bstRecord;

    edm::RefProd<L1AcceptBunchCrossing> l1AcceptBunchCrossingRef ;
    edm::RefProd<L1TriggerScalers> l1TriggerScalersRef ;
    edm::RefProd<L1TriggerRates> l1TriggerRatesRef ;
    edm::RefProd<Level1TriggerScalers> level1TriggerScalersRef ;
    edm::RefProd<Level1TriggerRates> level1TriggerRatesRef ;
    edm::RefProd<LumiScalers> lumiScalersRef ;
    edm::RefProd<DcsStatus> dcsStatusRef ;
    edm::RefProd<BeamSpotOnline> beamSpotOnlineRef ;
    edm::RefProd<BSTRecord> bstRecordRef ;

    L1AcceptBunchCrossingCollection l1AcceptBunchCrossingCollection;
    edm::Wrapper<L1AcceptBunchCrossingCollection> 
      w_l1AcceptBunchCrossingCollection;

    L1TriggerScalersCollection l1TriggerScalersCollection;
    edm::Wrapper<L1TriggerScalersCollection> w_l1TriggerScalersCollection;
 
    L1TriggerRatesCollection l1TriggerRatesCollection;
    edm::Wrapper<L1TriggerRatesCollection> w_l1TriggerRatesCollection;

    Level1TriggerScalersCollection level1TriggerScalersCollection;
    edm::Wrapper<Level1TriggerScalersCollection> w_level1TriggerScalersCollection;

    Level1TriggerRatesCollection level1TriggerRatesCollection;
    edm::Wrapper<Level1TriggerRatesCollection> w_level1TriggerRatesCollection;

    LumiScalersCollection lumiScalersCollection;
    edm::Wrapper<LumiScalersCollection> w_lumiScalersCollection;

    DcsStatusCollection dcsStatusCollection;
    edm::Wrapper<DcsStatusCollection> w_dcsStatusCollection;

    BeamSpotOnlineCollection beamSpotOnlineCollection;
    edm::Wrapper<BeamSpotOnlineCollection> w_beamSpotOnlineCollection;
  };
}
