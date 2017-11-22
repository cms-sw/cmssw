#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineBeamSpotRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace DataFormats_OnlineMetaData {
  struct dictionary {

    DCSRecord dcsRecord;
    OnlineBeamSpotRecord onlineBeamSpotRecord;
    OnlineLuminosityRecord onlineLuminosityRecord;

    edm::Wrapper<DCSRecord> w_dcsRecord;
    edm::Wrapper<OnlineBeamSpotRecord> w_onlineBeamSpotRecord;
    edm::Wrapper<OnlineLuminosityRecord> w_onlineLuminosityRecord;

    edm::RefProd<DCSRecord> dcsRecordRef;
    edm::RefProd<OnlineBeamSpotRecord> onlineBeamSpotRecordRef;
    edm::RefProd<OnlineLuminosityRecord> onlineLuminosityRecordRef;
  };
}
