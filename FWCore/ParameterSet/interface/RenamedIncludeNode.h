#ifndef ParameterSet_RenamedIncludeNode_h
#define ParameterSet_RenamedIncludeNode_h

#include "FWCore/ParameterSet/interface/IncludeNode.h"

/** IncludeNodes which are immediately renamed, so
 *  they aren't subject to restrictions on multiple inclusion
*/

namespace edm {
  namespace pset {

    class RenamedIncludeNode : public IncludeNode
    {
    public:
      RenamedIncludeNode(const std::string & type, const std::string & name, 
                         const std::string & targetType, const std::string & newName, 
                         const std::string & targetName, int line);

      virtual Node * clone() const { return new RenamedIncludeNode(*this);}

      /// some subclasses may wish to allow multiple includes
      virtual bool checkMultipleIncludes() const {return false;}

      virtual bool check(bool strict) const;

      virtual void print(std::ostream & out, PrintOptions options) const;

    private:
      /// only accept the named node.  Will rename it to newName_;
      bool checkTarget(NodePtr node) const;

      /// used for a check: the type of the subnode to be renamed
      std::string targetType_;
      std::string newName_;
      /// the name of the subnode to be renamed
      std::string targetName_;
    };

  }
}

#endif

