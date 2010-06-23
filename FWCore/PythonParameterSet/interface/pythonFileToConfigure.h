#ifndef FWCore_PythonParameterSet_pythonFileToConfigure_h
#define FWCore_PythonParameterSet_pythonFileToConfigure_h

/*
 *  pythonFileToConfigure.h
 *  CMSSW
 *
 *  Created by Chris Jones on 10/3/06.
 *
 */
#include <string>

namespace edm {
  std::string pythonFileToConfigure(const std::string& iPythonFileName);
}
#endif
