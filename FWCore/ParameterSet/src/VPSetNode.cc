#include "FWCore/ParameterSet/interface/VPSetNode.h"
#include "FWCore/ParameterSet/interface/PSetNode.h"
#include "FWCore/ParameterSet/interface/VEntryNode.h"
#include "FWCore/ParameterSet/interface/Nodes.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <ostream>
#include <iterator>
#include <iosfwd>

namespace edm {
  namespace pset {

    VPSetNode::VPSetNode(const std::string& typ,
                         const std::string& name,
                         NodePtrListPtr value,
                         bool untracked,
                         int line) :
      CompositeNode(name,value, line),
      type_(typ),
      tracked_(!untracked)
    { }

    std::string VPSetNode::type() const { return type_; }


    void VPSetNode::print(std::ostream& ost, Node::PrintOptions options) const
    {
      assert(nodes()!=0);

      ost << type() << " " << name() << " = {\n";
      if(!nodes()->empty())
        {
          //copy_all(*value_, std::ostream_iterator<NodePtr>(ost,",\n  "));
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


    void VPSetNode::resolveUsingNodes(const NodeMap & blocks, bool strict)
    {
      // if a node is just a std::string, find the block it refers to
      NodePtrList::iterator nodeItr(nodes_->begin()),e(nodes_->end());
      for(;nodeItr!=e;++nodeItr)
      {
        if((**nodeItr).type() == "string")
        {
          // find the block
          std::string blockName = (**nodeItr).name();
          NodeMap::const_iterator blockPtrItr = blocks.find(blockName);
          if(blockPtrItr == blocks.end()) {
             throw edm::Exception(errors::Configuration,"")
               << "VPSet: Cannot find parameter block " << blockName
               << "\nfrom " << traceback();
          }

          //@@ will this destruct the old entry correctly?
          // doesn't deep-copy
          *nodeItr = blockPtrItr->second;
        }
        else
        {
          // if it isn't a using node, check the grandchildren
          (**nodeItr).resolveUsingNodes(blocks, strict);
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
      pset.insert(false, name(), makeEntry());
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
      
      return Entry(name(), sets, tracked_); 
    }


    void VPSetNode::replaceWith(const ReplaceNode * replaceNode)
    {
      // see if it's a replace or an append
      if(replaceNode->type() == "replace")
      {
        VPSetNode * replacement = replaceNode->value<VPSetNode>();

        if(replacement != 0) 
        {
          nodes_ = replacement->nodes_;
        }
        else 
        {
          // maybe it's a blank VEntryNode
          VEntryNode * ventryNode = replaceNode->value<VEntryNode>();
          if(ventryNode != 0 && ventryNode->value()->empty())
          {
            nodes_->clear();
          }
          else 
          {
            throw edm::Exception(errors::Configuration)
              << "Cannot replace entry vector " << name()
              <<   " with " << replaceNode->type()
              << "\nfrom " << traceback();
          }
        }
      }
      else if(replaceNode->type() == "replaceAppend")
      {
        append(replaceNode->value());
      }
      else
      {
         throw edm::Exception(errors::Configuration)
            << "Cannot replace entry vector" << name()
            <<   " with " << replaceNode->type()
            << "\nfrom " << traceback();
      }

      setModified(true);
    }


    void VPSetNode::append(NodePtr ptr)
    {
      // single or multiple?  ContentsNodes never say their type.
      // they represent a single PSet
      if(ptr->type() == "")
      {
        nodes_->push_back(ptr);
      }
      else if(ptr->type() == "PSet")
      {
        // make a ContentsNode from this PSetNode
        PSetNode * psetNode =  dynamic_cast<PSetNode*>(ptr.get());
        NodePtr n(new ContentsNode(psetNode->nodes(), psetNode->line()));
        nodes_->push_back(n);
      }
      else
      {
        // try VPSet
        VPSetNode * vpsetNode =  dynamic_cast<VPSetNode*>(ptr.get());
        if(vpsetNode != 0)
        {
          NodePtrListPtr entries = vpsetNode->nodes_;
          for(NodePtrList::const_iterator itr = entries->begin(), itrEnd = entries->end();
              itr != itrEnd; ++itr)
          {
            nodes_->push_back(*itr);
          }
        }
        // neither Entry or VPSet
        else
        {
          throw edm::Exception(errors::Configuration)
            << "Bad type to append to VPSet "
            <<  ptr->type();
        }
      }
    }


  }
}
