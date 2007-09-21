#include "FWCore/ParameterSet/interface/Node.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

namespace edm {

  namespace pset {

    Node::Node(std::string const& n, int li) 
    : name_(n), 
      line_(li), 
      modified_(false),
      cloned_(false),
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
          << "No way to replace node " << name()
          << "\nfrom " << traceback();
    }


    bool Node::isInclude() const
    {
      return (type().substr(0,7) == "include");
    }


    const Node * Node::getIncludeParent() const
    {
      const Node * result = 0; 
      const Node * currentNode = this; 
      bool done = false;
      while(!done)
      {
        currentNode = currentNode->getParent();
        if(currentNode == 0)
        {
          done = true;
        }
        else 
        {
          if(currentNode->isInclude())
          {
            result = currentNode;
            done = true;
          }
        }
      }
      return result;
    }
          

    std::string Node::includeParentSuffix() const
    {
      const Node * includeParent = getIncludeParent();
      if(includeParent == 0) return "";

      int nletters = includeParent->name().length();
      assert(nletters >= 3);
      return includeParent->name().substr(nletters-3);
    }
    
 
    void Node::dotDelimitedPath(std::string & path) const
    {
      if(!path.empty())
      {
        path.insert(0, ".");
      }
      path.insert(0, name());

      // add parents, if any
      if(parent_ != 0)
      {
        parent_->dotDelimitedPath(path);
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


    std::string Node::traceback() const
    {
      std::ostringstream tr;
      printTrace(tr);
      std::string result = tr.str();
      if(result.empty()) result = "<MAIN CFG>";
      return result;
    }

   
    void Node::locate(const std::string & s, std::ostream & out) const
    {
      if(name().find(s,0) != std::string::npos) 
      {
        print(out);
        out << "\n";
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
