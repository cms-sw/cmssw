#ifndef ParameterSet_IncludeNode_h
#define ParameterSet_IncludeNode_h

#include "FWCore/ParameterSet/interface/CompositeNode.h"

/** IncludeNodes contain a FileInPath
    for the file to be included.  They are
    resolved through a second pass through the parser
*/

namespace edm {
  namespace pset {

    class IncludeNode : public CompositeNode
    {
    public:
      IncludeNode(const std::string & type, const std::string & name, int line);

      virtual std::string type() const {return type_;}
      virtual Node * clone() const { return new IncludeNode(*this);}
      virtual void accept(Visitor& v) const;

      /// fills in the CompositeNode by parsing the included file
      /// the argument prevents circular includes
      void resolve(std::list<std::string> & openFiles);

      /// makes sure that this is not a currently-open file 
      void checkCircularInclusion(const std::list<std::string> & openFiles) const;
    private:
      std::string type_;

    };

  }
}

#endif

