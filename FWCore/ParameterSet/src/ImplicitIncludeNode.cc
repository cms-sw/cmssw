#include "FWCore/ParameterSet/interface/ImplicitIncludeNode.h"
#include "FWCore/ParameterSet/interface/IncludeFileFinder.h"

namespace edm {
  namespace pset {

    ImplicitIncludeNode::ImplicitIncludeNode(const std::string & moduleClass, 
                                             const std::string & moduleLabel, int line)
    : IncludeNode("implicitInclude", "", line),
    moduleClass_(moduleClass),
    moduleLabel_(moduleLabel)
    {
    }


    void ImplicitIncludeNode::resolve(std::list<std::string> & openFiles,
                                       std::list<std::string> & sameLevelIncludes)
    {
      // wish there were a way to initialize this only once
      IncludeFileFinder finder;
      FileInPath file = finder.find(moduleClass_, moduleLabel_);

      // going back to relative path, for coding convenience
      setName(file.relativePath());
      IncludeNode::resolve(openFiles, sameLevelIncludes);
    }

  }
}

