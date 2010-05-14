
#include "CondCore/DTPlugins/interface/DTConfigPluginHandler.h"

class DTConfigHandlerFactory {

 public:

  DTConfigHandlerFactory() {
    DTConfigPluginHandler::build();
  }

};

static DTConfigHandlerFactory dtchf;

