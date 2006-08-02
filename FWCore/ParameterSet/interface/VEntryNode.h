#ifndef ParameterSet_VEntryNode_h
#define ParameterSet_VEntryNode_h

#include "FWCore/ParameterSet/interface/Node.h"

    /**
      -----------------------------------------
      VEntries hold: vint32, vuint32, vdouble, vstring
    */



namespace edm {
  namespace pset {


    struct VEntryNode : public Node
    {
      VEntryNode(const std::string& typ, const std::string& name,
                 StringListPtr values,bool tracked, int line=-1);
      /// deep copy
      VEntryNode(const VEntryNode & n);
      virtual Node * clone() const { return new VEntryNode(*this);}

      virtual std::string type() const;
      virtual void print(std::ostream& ost, PrintOptions options) const;

      virtual void accept(Visitor& v) const;
      // keeps the orignal type and tracked-ness
      virtual void replaceWith(const ReplaceNode *);
      /// append a node into the vector
      virtual void append(NodePtr ptr);

      virtual edm::Entry makeEntry() const;


      std::string type_;
      StringListPtr value_;
      bool tracked_;
    };

  }
}

#endif

