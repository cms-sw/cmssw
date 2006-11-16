#include "FWCore/ParameterSet/interface/WrapperNode.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/Visitor.h"

using std::string;


namespace edm {
  namespace pset {

    WrapperNode::WrapperNode(const string& type, const string& name,
                             NodePtr w,int line):
      Node(name, line),
      type_(type),
      wrapped_(w)
    { }

    string WrapperNode::type() const { return type_; }


    bool WrapperNode::findChild(const std::string & childName, NodePtr & result)
    {
      bool found = false;
      // first check the wrapped node
      if(wrapped()->name() == childName)
      {
        result = wrapped();
        found = true;
      }
      // be transparent.
      else 
      { 
        found = wrapped()->findChild(childName, result);
      }
      return found;
    }


    void WrapperNode::print(std::ostream& ost, Node::PrintOptions options) const
    {
      ost << type() << " " << name() << " = {\n"
          << wrapped_
          << "\n}\n";
    }

    void WrapperNode::accept(Visitor& v) const
    {
      // we do not visit lower module here, the scheduler uses those
      v.visitWrapper(*this);
    }


    void WrapperNode::insertInto(edm::ProcessDesc & procDesc) const
    {
      boost::shared_ptr<WrapperNode> wrapperClone(new WrapperNode(*this));
      procDesc.addPathFragment(wrapperClone);
    }

  }
}

