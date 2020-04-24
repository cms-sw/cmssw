#include "DataFormats/TCDS/interface/TCDSRecord.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace DataFormats_TCDS {
  struct dictionary {

    TCDSRecord tcdsRecord;

    edm::Wrapper<TCDSRecord> w_tcdsRecord;

    edm::RefProd<TCDSRecord> tcdsRecordRef;
  };
}
