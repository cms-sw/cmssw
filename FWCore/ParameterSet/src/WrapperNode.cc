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

    void WrapperNode::print(std::ostream& ost) const
    {
      ost << type_ << " " << name << " = {\n"
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

