#include "FWCore/ParameterSet/interface/Node.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <algorithm>

#include <iostream>

namespace edm {

  namespace pset {

    Node::Node(std::string const& n, int li) 
    : name_(n), 
      line_(li), 
      modified_(false),
      parent_(0)
    { 
    }


    Node::~Node() { }

    void Node::setModified(bool value) 
    {
      modified_ = value;
    }

    void Node::replaceWith(const ReplaceNode *) {
       throw edm::Exception(errors::Configuration)
          << "No way to replace node " << name();
    }


    void Node::assertNotModified() const
    {
      if(isModified()) {
       throw edm::Exception(errors::Configuration)
          << "Cannot replace a node that has already been modified: " << name();
      }
    }


    void Node::printTrace(std::ostream & out) const 
    {
      // default behavior is to pass the message up to the parent,
      // in case the parent knows what to do.
      if(parent_ != 0) 
      {
        parent_->printTrace(out);
      }
    }     

   
    void Node::locate(const std::string & s, std::ostream & out) const
    {
      if(name().find(s,0) != std::string::npos) 
      {
        out << "Found " << name() << "\n";
        printTrace(out);
        out << "\n";
      }
    }


    void Node::insertInto(edm::ParameterSet & pset) const
    {
      pset.insert(false, name(), makeEntry());
    }


    void Node::insertInto(edm::ProcessDesc & procDesc) const
    {
      // don't want to make this method pure-virtual, so I'll settle
      // for a runtime error if not overloaded
      throw edm::Exception(errors::Configuration)
         << "No way to convert this node, " <<  name()
         << ", to a ProcessDesc Entry";
    }


    edm::Entry Node::makeEntry() const
    {
      // don't want to make this method pure-virtual, so I'll settle
      // for a runtime error if not overloaded
      throw edm::Exception(errors::Configuration)
         << "No way to convert this node, " <<  name()
         << ", to a ParameterSet Entry";
    }


  }
}
