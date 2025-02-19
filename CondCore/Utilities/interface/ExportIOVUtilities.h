#include "CondCore/Utilities/interface/Utilities.h"
#include <string>

namespace cond {

  class ExportIOVUtilities : public Utilities {
    public:
       explicit ExportIOVUtilities(std::string const& name );
       ~ExportIOVUtilities();
      int execute();
  };

}

