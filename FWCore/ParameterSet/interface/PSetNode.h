#ifndef ParameterSet_PSetNode_h
#define ParameterSet_PSetNode_h

#include "FWCore/ParameterSet/interface/CompositeNode.h"

namespace edm {
  namespace pset {

    struct PSetNode : public CompositeNode
    {
      PSetNode(const std::string& typ,
               const std::string& name,
               NodePtrListPtr value,
               bool tracked,
               int line=-1);
      virtual Node * clone() const { return new PSetNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;
      virtual bool isModified() const;

      virtual void accept(Visitor& v) const;
      virtual void replaceWith(const ReplaceNode * replaceNode);

      std::string type_;
      bool tracked_;
    };

  }
}

#endif

