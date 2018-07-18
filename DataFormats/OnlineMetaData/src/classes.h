#include "DataFormats/OnlineMetaData/interface/CTPPSRecord.h"
#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace DataFormats_OnlineMetaData {
  struct dictionary {

    CTPPSRecord ctppsRecord;
    DCSRecord dcsRecord;
    OnlineLuminosityRecord onlineLuminosityRecord;

    edm::Wrapper<CTPPSRecord> w_ctppsRecord;
    edm::Wrapper<DCSRecord> w_dcsRecord;
    edm::Wrapper<OnlineLuminosityRecord> w_onlineLuminosityRecord;

    edm::RefProd<CTPPSRecord> ctppsRecordRef;
    edm::RefProd<DCSRecord> dcsRecordRef;
    edm::RefProd<OnlineLuminosityRecord> onlineLuminosityRecordRef;
  };
}
