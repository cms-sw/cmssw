#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  namespace pset {

    ReplaceNode::ReplaceNode(const ReplaceNode & n)
    : Node(n),
      type_(n.type()),
      value_( NodePtr(n.value_->clone()) )
    {
    }


    bool ReplaceNode::isEmbedded() const
    {
      return (getParent()->isInclude());
    }
 
    void ReplaceNode::print(std::ostream& ost, Node::PrintOptions options) const
    {
      value_->print(ost, options);
    }

    void ReplaceNode::accept(Visitor& v) const
    {
      throw edm::Exception(errors::Configuration,"Replace Nodes should always be processed by the postprocessor.  Please contact an EDM developer") << "\nfrom " << traceback();
    }

  }
}
