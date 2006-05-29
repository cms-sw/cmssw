#ifndef ParameterSet_VPSetNode_h
#define ParameterSet_VPSetNode_h

#include "FWCore/ParameterSet/interface/CompositeNode.h"

namespace edm {
  namespace pset {

    /*
      -----------------------------------------
      VPSets hold: ParameterSet nodes or ParameterSet names/IDs stored in Entries
    */

    struct VPSetNode : public CompositeNode
    {
      VPSetNode(const std::string& typ,
                const std::string& name,
                NodePtrListPtr value,
                bool tracked,
                int line=-1);
      virtual Node * clone() const { return new VPSetNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;

      virtual void accept(Visitor& v) const;

      /// when a sub-node is a StringNode, find the PSet
      /// it refers to
      virtual void resolveUsingNodes(const NodeMap & blocks);

      /// Nodes which can exist on the top level of the
      /// parse tree should implement this
      virtual void insertInto(ProcessDesc & procDesc) const;
      /// Nodes which can be inserted into ParameterSets
      /// which aren't top-level processes should overload this.
      virtual void insertInto(edm::ParameterSet & pset) const;
      /// makes a ParameterSet Entry for this Node
      virtual edm::Entry makeEntry() const;

      std::string type_;
      bool tracked_;
    };

  }
}

#endif

