#include "CondFormats/BTagObjects/src/headers.h"



namespace CondFormats_BTagObjects {
  struct dictionary {
    std::vector<float> b1;

    BTagEntry bte1;
    BTagEntry::Parameters bte_p1;
    std::vector<BTagEntry> v_bte1;
    std::map<std::string, std::vector<BTagEntry> > mv_bte1;
    BTagCalibration btc1;

  };
}
