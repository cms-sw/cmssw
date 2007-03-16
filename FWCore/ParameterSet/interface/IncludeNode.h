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

     /// IncludeNodes don't add their names to the path
      virtual void dotDelimitedPath(std::string & path) const;

      virtual void print(std::ostream & ost, PrintOptions options) const;

      /// prints file and line number, and passes up
      virtual void printTrace(std::ostream & ost) const;
      
      /// fills in the CompositeNode by parsing the included file
      /// the argument prevents circular includes
      virtual void resolve(std::list<std::string> & openFiles,
                           std::list<std::string> & sameLevelIncludes);

      /// some subclasses may wish to allow multiple includes
      virtual bool checkMultipleIncludes() const {return true;}
      
      /// simply inserts all subnodes
      virtual void insertInto(edm::ProcessDesc & procDesc) const;

      std::string fullPath() const {return fullPath_;}
      bool isResolved() const {return isResolved_;}

    private:
      //throws an exception if more than one module defined in a .cfi
      bool check() const;

      std::string type_;
      std::string fullPath_;
      bool isResolved_;
    };

  }
}

#endif

