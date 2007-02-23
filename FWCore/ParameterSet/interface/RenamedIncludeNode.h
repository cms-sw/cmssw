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

      /// only accept the named node.  Will rename it to newName_;
      virtual bool checkTarget(NodePtr node);

      /// adds the filtering
      virtual void resolve(std::list<std::string> & openFiles,
              std::list<std::string> & sameLevelIncludes);


      virtual void print(std::ostream & out, PrintOptions options) const;

    private:
      /// used for a check: the type of the subnode to be renamed
      std::string targetType_;
      std::string newName_;
      /// the name of the subnode to be renamed
      std::string targetName_;
    };

  }
}

#endif

