#include "DataFormats/TCDS/interface/TCDSRecord.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

TCDSRecord tcdsRecord;

namespace DataFormats_TCDS {
  struct dictionary {

    edm::Wrapper<TCDSRecord> w_tcdsRecord;

    edm::RefProd<TCDSRecord> tcdsRecordRef;
  };
}
