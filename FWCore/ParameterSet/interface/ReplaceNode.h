#ifndef ParameterSet_ReplaceNode_h
#define ParameterSet_ReplaceNode_h

#include "FWCore/ParameterSet/interface/Node.h"
#include <iosfwd>

namespace edm {
  namespace pset {

    struct Visitor;

    /**
      -----------------------------------------
      Replace : instructions for replacing the value of this node
    */

    struct ReplaceNode : public Node
    {
      ReplaceNode(const std::string & type, const std::string& name,
                  NodePtr value, int line=-1)
      : Node(name, line), type_(type), value_(value) {}
      /// deep copy
      ReplaceNode(const ReplaceNode & n);
      virtual Node * clone() const { return new ReplaceNode(*this);}
      virtual std::string type() const {return type_;}
      virtual void print(std::ostream& ost, PrintOptions options) const;
      virtual void accept(Visitor& v) const;
      NodePtr value() const {return value_;}

      std::string type_;
      NodePtr value_;
    };

  }
}

#endif

