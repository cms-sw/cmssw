#include "FWCore/ParameterSet/interface/OperandNode.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
namespace edm {
  namespace pset {


    //--------------------------------------------------
    // OperandNode
    //--------------------------------------------------

    OperandNode::OperandNode(const std::string& type,
                             const std::string& name,
                             int line):
      Node(name, line),
      type_(type)
    {  }

    std::string OperandNode::type() const { return type_; }

    void OperandNode::print(std::ostream& ost, Node::PrintOptions options) const
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
