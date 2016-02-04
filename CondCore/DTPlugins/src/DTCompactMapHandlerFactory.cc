
#include "CondCore/DTPlugins/interface/DTCompactMapPluginHandler.h"

class DTCompactMapHandlerFactory {

 public:

  DTCompactMapHandlerFactory() {
    DTCompactMapPluginHandler::build();
  }

};

static DTCompactMapHandlerFactory dtcmhf;

