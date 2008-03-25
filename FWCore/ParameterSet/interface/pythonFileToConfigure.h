#if !defined(FWCore_Framework_pythonToConfigure_h)
#define FWCore_Framework_pythonToConfigure_h
/*
 *  pythonToConfigure.h
 *  CMSSW
 *
 *  Created by Chris Jones on 10/3/06.
 *  Copyright 2006 __MyCompanyName__. All rights reserved.
 *
 */
#include <string>

namespace edm {
  std::string pythonFileToConfigure(const std::string& iPythonFileName);
}
#endif /* defined(FWCore_Framework_pythonToConfigure_h) */

