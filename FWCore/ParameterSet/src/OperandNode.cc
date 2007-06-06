#include "FWCore/ParameterSet/interface/OperandNode.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
using namespace std;

namespace edm {
  namespace pset {


    //--------------------------------------------------
    // OperandNode
    //--------------------------------------------------

    OperandNode::OperandNode(const string& type,
                             const string& name,
                             int line):
      Node(name, line),
      type_(type)
    {  }

    string OperandNode::type() const { return type_; }

    void OperandNode::print(ostream& ost, Node::PrintOptions options) const
    {
      ost << name();
    }

    void OperandNode::accept(Visitor& v) const
    {
      v.visitOperand(*this);
      //throw runtime_error("OperatandNodes cannot be visited");
    }
  }
}
