#ifndef ParameterSet_WrapperNode_h
#define ParameterSet_WrapperNode_h

#include "FWCore/ParameterSet/interface/Node.h"

namespace edm {
  namespace pset {

    /*
      -----------------------------------------
      Wrappers hold: sequences, paths, endpaths
      They hold another Node that actually contains the information.
    */

    class WrapperNode : public Node
    {
    public:
      WrapperNode(const std::string& type, const std::string& name,
                  NodePtr w, int line=-1);
      virtual Node * clone() const { return new WrapperNode(*this);}

      virtual std::string type() const;
      NodePtr & wrapped() {return wrapped_;}
      const NodePtr & wrapped() const {return wrapped_;}

      virtual void print(std::ostream& ost, PrintOptions options) const;

      virtual void accept(Visitor& v) const;

      /// Insert into the top level of the tree
      /// the ProcessDescription has a separate
      /// field for the Wrapper nodes
      virtual void insertInto(edm::ProcessDesc & procDesc) const;

    private:
      std::string type_;
      NodePtr wrapped_;
    };


  }
}

#endif

