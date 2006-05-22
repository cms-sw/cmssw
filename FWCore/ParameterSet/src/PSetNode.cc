#include "FWCore/ParameterSet/interface/PSetNode.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"

using namespace std;

namespace edm {

  namespace pset {


    PSetNode::PSetNode(const string& t,
                       const string& n,
                       NodePtrListPtr v,
                       bool tracked,
                       int line) :
      CompositeNode(n, v, line),
      type_(t),
      tracked_(tracked)
    {}


    string PSetNode::type() const { return type_; }

    void PSetNode::print(ostream& ost) const
    {
      // if(!name.empty())
      ost << type_ << " " << name << " = ";

      CompositeNode::print(ost);
    }

    void PSetNode::accept(Visitor& v) const
    {
      v.visitPSet(*this);
    }


    bool PSetNode::isModified() const 
    {
      return modified_ || CompositeNode::isModified();
    }

    void PSetNode::replaceWith(const ReplaceNode * replaceNode)
    {
      assertNotModified();
      NodePtr replacementPtr = replaceNode->value_;
      PSetNode * replacement = dynamic_cast<PSetNode*>(replacementPtr.get());
      assert(replacement != 0);

      nodes_ = replacement->nodes_;
      setModified(true);
    }

  }
}
