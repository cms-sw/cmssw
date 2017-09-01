#include "DataFormats/TCDS/interface/TCDSRecord.h"

TCDSRecord tcdsRecord;

namespace DataFormats_TCDS {
  struct dictionary {

    edm::Wrapper<TCDSRecord> w_tcdsRecord;

    edm::RefProd<TCDSRecord> tcdsRecordRef;
  }
}
