#include "FWCore/ParameterSet/interface/PSetNode.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/VEntryNode.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <ostream>
#include <iostream>

namespace edm {

  namespace pset {


    PSetNode::PSetNode(const std::string& t,
                       const std::string& n,
                       NodePtrListPtr v,
                       bool untracked,
                       int line) :
      CompositeNode(n, v, line),
      type_(t),
      tracked_(!untracked)
    {
      // processes shouldn't have minuses in their names
      if(type() == "process" && n.find("-",0) != std::string::npos)
      {
        throw edm::Exception(errors::Configuration,"PSetError")
           << "Process names should not have minus signs in them.\n"
           << "It will lead to a ROOT error.";
      } 
    }


    std::string PSetNode::type() const { return type_; }


    void PSetNode::dotDelimitedPath(std::string & path) const
    {
      // processes don't add their names to the path
      if(type() != "process")
      {
        CompositeNode::dotDelimitedPath(path);
      }
    }


    void PSetNode::print(std::ostream& ost, Node::PrintOptions options) const
    {
      const char* t = tracked_ ? "" : "untracked ";
      ost << t << type() << " " << name() << " = ";

      CompositeNode::print(ost, options);
    }

    void PSetNode::accept(Visitor& v) const
    {
      v.visitPSet(*this);
    }


    bool PSetNode::isModified() const 
    {
      return Node::isModified() || CompositeNode::isModified();
    }


    void PSetNode::replaceWith(const ReplaceNode * replaceNode)
    {
      PSetNode * replacement = replaceNode->value<PSetNode>();
      if(replacement != 0)
      {
        nodes_ = replacement->nodes_;
      }
      else
      {
        // maybe it's an empty {}, interpretedf as a VEntry
        VEntryNode * entries  = replaceNode->value<VEntryNode>();
        if(entries == 0 || entries->value()->size() != 0) 
        {
           throw edm::Exception(errors::Configuration,"PSetError")
            << " Bad replace for PSet " << name()
            << "\nIn " << traceback() << std ::endl;
        }
        else {
          nodes_->clear();
        }
      }
      setModified(true);
    }


    edm::Entry PSetNode::makeEntry() const
    {
      // verify that this ia not a process related node
      if(type()=="process")
        {
          throw edm::Exception(errors::Configuration,"PSetError")
            << "ParameterSet: problem with making a parameter set.\n"
            << "Attempt to convert process input to ParameterSet";
        }

      boost::shared_ptr<ParameterSet> psetPtr(new ParameterSet);
      // do the subnodes
      CompositeNode::insertInto(*psetPtr);
      return Entry(name(), *psetPtr, tracked_);
    }

    void PSetNode::insertInto(edm::ParameterSet & pset) const
    {
      pset.insert(false, name(), makeEntry());
    }



    void PSetNode::insertInto(edm::ProcessDesc & procDesc) const
    {
      insertInto(*(procDesc.getProcessPSet()));
    }


    void PSetNode::fillProcess(edm::ProcessDesc & procDesc) const
    {
      if(type()!="process")
      {
        throw edm::Exception(errors::Configuration,"PSetError")
          << "ParameterSet: problem with making a parameter set.\n"
          << "Attempt to make a ProcessDesc with a PSetNode which is not a process"
          << "\nIn " << traceback();
      }

      procDesc.getProcessPSet()->addParameter("@process_name", name());
      // insert the subnodes as top-level nodes
      NodePtrList::const_iterator i(nodes()->begin()),e(nodes()->end());
      for(;i!=e;++i)
      {
        try
        {
          (**i).insertInto(procDesc);
        }
        catch(edm::Exception & e)
        {
          // print some extra debugging
          std::ostringstream message;
          message << "\nIn variable " << (**i).name() 
                  << "\nIncluded from:\n" << (**i).traceback();
          e.append(message.str());
         
          // pass it on(errors::Configuration
          throw e;
        }
      }

    }


  }
}
