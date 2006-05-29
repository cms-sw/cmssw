#include "FWCore/ParameterSet/interface/Node.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  namespace pset {

    Node::~Node() { }

    void Node::replaceWith(const ReplaceNode *) {
       throw edm::Exception(errors::Configuration)
          << "No way to replace node " << name;
    }


    void Node::assertNotModified() const
    {
      if(isModified()) {
       throw edm::Exception(errors::Configuration)
          << "Cannot replace a node that has already been modified: " << name;
      }
    }


    void Node::insertInto(edm::ParameterSet & pset) const
    {
      pset.insert(false, name, makeEntry());
    }


    void Node::insertInto(edm::ProcessDesc & procDesc) const
    {
      // don't want to make this method pure-virtual, so I'll settle
      // for a runtime error if not overloaded
      throw edm::Exception(errors::Configuration)
         << "No way to convert this node, " <<  name
         << ", to a ProcessDesc Entry";
    }


    edm::Entry Node::makeEntry() const
    {
      // don't want to make this method pure-virtual, so I'll settle
      // for a runtime error if not overloaded
      throw edm::Exception(errors::Configuration)
         << "No way to convert this node, " <<  name
         << ", to a ParameterSet Entry";
    }


  }
}
