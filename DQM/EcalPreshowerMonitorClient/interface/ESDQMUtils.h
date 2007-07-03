#ifndef ESDQMUtils_H
#define ESDQMUtils_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementT.h"

class ESDQMUtils {

 public:

  static void resetME( const MonitorElement* me );

 protected:

  ESDQMUtils() { }

};

#endif


