#ifndef ParameterSet_ImplicitIncludeNode_h
#define ParameterSet_ImplicitIncludeNode_h

/** this type of node accepts a class name and
 *  a label, and tries to find the correct include file
 */

#include "FWCore/ParameterSet/interface/IncludeNode.h"

namespace edm {
  namespace pset {
  
    class ImplicitIncludeNode : public IncludeNode
    {
    public:
      ImplicitIncludeNode(const std::string & moduleClass,
                          const std::string & moduleLabel, int line);

      void resolve(std::list<std::string> & openFiles,
                   std::list<std::string> & sameLevelIncludes);

    private:
      std::string moduleClass_;
      std::string moduleLabel_;
    };

  }
}

#endif

