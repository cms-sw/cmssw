#include "FWCore/ParameterSet/interface/Node.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace std;

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
  }
}
