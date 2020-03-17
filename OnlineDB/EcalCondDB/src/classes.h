#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

namespace OnlineDB_EcalCondDB {
  struct dictionnary {
    RunTag r1;
    RunIOV r2;
    EcalCondDBInterface e1;
  };
}
