#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <ostream>

namespace edm {
  namespace pset {

    ReplaceNode::ReplaceNode(const ReplaceNode & n)
    : Node(n),
      type_(n.type()),
      value_( NodePtr(n.value_->clone()) )
    {
    }


    void ReplaceNode::print(std::ostream& ost) const
    {
      value_->print(ost);
    }

    void ReplaceNode::accept(Visitor& v) const
    {
      throw edm::Exception(errors::LogicError,"Replace Nodes should always be processed by the postprocessor.  Please contact an EDM developer");
    }

  }
}
