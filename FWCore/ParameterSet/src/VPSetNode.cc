#include "FWCore/ParameterSet/interface/VPSetNode.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <ostream>
#include <iterator>
using namespace std;

namespace edm {
  namespace pset {

    VPSetNode::VPSetNode(const string& typ,
                         const string& name,
                         NodePtrListPtr value,
                         bool tracked,
                         int line) :
      CompositeNode(name,value, line),
      type_(typ),
      tracked_(tracked)
    { }

    string VPSetNode::type() const { return type_; }


    void VPSetNode::print(ostream& ost, Node::PrintOptions options) const
    {
      assert(nodes()!=0);

      ost << type() << " " << name << " = {\n";
      if(!nodes()->empty())
        {
          //copy(value_->begin(),value_->end(),
          //   ostream_iterator<NodePtr>(ost,",\n  "));
          NodePtrList::const_iterator ie(nodes()->end()),ib(nodes()->begin());
          --ie;
          copy(ib,ie,
               std::ostream_iterator<NodePtr>(ost,", "));
          ost << *ie;
        }  ost << "\n}\n";

    }

    void VPSetNode::accept(Visitor& v) const
    {
      v.visitVPSet(*this);
    }


    void VPSetNode::resolveUsingNodes(const NodeMap & blocks)
    {
      // if a node is just a string, find the block it refers to
      NodePtrList::iterator nodeItr(nodes_->begin()),e(nodes_->end());
      for(;nodeItr!=e;++nodeItr)
      {
        if((**nodeItr).type() == "string")
        {
          // find the block
          string blockName = (**nodeItr).name;
          NodeMap::const_iterator blockPtrItr = blocks.find(blockName);
          if(blockPtrItr == blocks.end()) {
             throw edm::Exception(errors::Configuration,"")
               << "Cannot find parameter block " << blockName;
          }

          //@@ will this destruct the old entry correctly?
          // doesn't deep-copy
          *nodeItr = blockPtrItr->second;
        }
        else
        {
          // if it isn't a using node, check the grandchildren
          (**nodeItr).resolveUsingNodes(blocks);
        }
      } // loop over subnodes
    }


      /// Nodes which can exist on the top level of the
      /// parse tree should implement this
    void VPSetNode::insertInto(ProcessDesc & procDesc) const
    {
      insertInto(*(procDesc.getProcessPSet()));
    }

      /// Nodes which can be inserted into ParameterSets
      /// which aren't top-level processes should overload this.
    void VPSetNode::insertInto(edm::ParameterSet & pset) const 
    {
      pset.insert(false, name, makeEntry());
    }

      /// makes a ParameterSet Entry for this Node
    edm::Entry VPSetNode::makeEntry() const
    {
      std::vector<ParameterSet> sets;
      NodePtrList::const_iterator ie(nodes()->end()),ib(nodes()->begin());

      for( ; ib != ie; ++ib)
      {
        // make a ParameterSet for this PSetNode
        boost::shared_ptr<ParameterSet> pset(new ParameterSet);
        (**ib).insertInto(*pset);
        sets.push_back(*pset);
      }
      
      return Entry(sets, !tracked_); 
    }


    void VPSetNode::replaceWith(const ReplaceNode * replaceNode)
    {
      assertNotModified();
      NodePtr replacementPtr = replaceNode->value_;
      VPSetNode * replacement = dynamic_cast<VPSetNode*>(replacementPtr.get());
      assert(replacement != 0);

      nodes_ = replacement->nodes_;
      setModified(true);

    }

  }
}
