#include "FWCore/ParameterSet/interface/PSetNode.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/Utilities/interface/EDMException.h"


using namespace std;

namespace edm {

  namespace pset {


    PSetNode::PSetNode(const string& t,
                       const string& n,
                       NodePtrListPtr v,
                       bool tracked,
                       int line) :
      CompositeNode(n, v, line),
      type_(t),
      tracked_(tracked)
    {}


    string PSetNode::type() const { return type_; }

    void PSetNode::print(ostream& ost) const
    {
      // if(!name.empty())
      ost << type_ << " " << name << " = ";

      CompositeNode::print(ost);
    }

    void PSetNode::accept(Visitor& v) const
    {
      v.visitPSet(*this);
    }


    bool PSetNode::isModified() const 
    {
      return modified_ || CompositeNode::isModified();
    }


    void PSetNode::replaceWith(const ReplaceNode * replaceNode)
    {
      assertNotModified();
      NodePtr replacementPtr = replaceNode->value_;
      PSetNode * replacement = dynamic_cast<PSetNode*>(replacementPtr.get());
      assert(replacement != 0);

      nodes_ = replacement->nodes_;
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
      return Entry(*psetPtr, !tracked_);
    }

    void PSetNode::insertInto(edm::ParameterSet & pset) const
    {
      pset.insert(false, name, makeEntry());
    }



    void PSetNode::insertInto(edm::ProcessDesc & procDesc) const
    {
      procDesc.getProcessPSet()->insert(false, name, makeEntry());
    }


    void PSetNode::fillProcess(edm::ProcessDesc & procDesc) const
    {
      if(type()!="process")
      {
        throw edm::Exception(errors::Configuration,"PSetError")
          << "ParameterSet: problem with making a parameter set.\n"
          << "Attempt to make a ProcessDesc with a PSetNode which is not a process";
      }

      procDesc.getProcessPSet()->insert(true, "@process_name", edm::Entry(name, true));
      // insert the subnodes as top-level nodes
      NodePtrList::const_iterator i(nodes()->begin()),e(nodes()->end());
      for(;i!=e;++i)
      {
         (**i).insertInto(procDesc);
      }

    }


  }
}
