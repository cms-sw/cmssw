#include "FWCore/ParameterSet/interface/CompositeNode.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <ostream>
#include <iterator>

#include <iostream>

using namespace std;

namespace edm {

  namespace pset {

    CompositeNode::CompositeNode(const CompositeNode & n)
    : Node(n),
      nodes_(new NodePtrList)
    {
      NodePtrList::const_iterator i(n.nodes_->begin()),e(n.nodes_->end());
      for(;i!=e;++i)
      {
         nodes_->push_back( NodePtr((**i).clone()) );
      }
    }


    void CompositeNode::acceptForChildren(Visitor& v) const
    {
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(;i!=e;++i)
        {
          (*i)->accept(v);
        }
    }


    void CompositeNode::print(ostream& ost) const
    {
      ost << "  {\n  ";
      copy(nodes_->begin(),nodes_->end(),
           ostream_iterator<NodePtr>(ost,"\n  "));
      ost << "}\n";
    }


    void CompositeNode::setModified(bool value) 
    {
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(;i!=e;++i)
      {
        (*i)->setModified(value);
      }
    }


    bool CompositeNode::isModified() const 
    {
      // see if any child is modified
      bool result = modified_;
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(;(i!=e) &&  !result;++i)
      {
        result = result || (*i)->isModified();
      }
      return result;
    }


    NodePtr CompositeNode::findChild(const string & child)
    {
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(;i!=e;++i)
      {
         if((*i)->name == child) {
           return *i;
         }
      }

      // uh oh.  Didn't find it.
      throw edm::Exception(errors::Configuration)
        << "Cannot find child " << child
        << " in composite node " << name;
    }


    void CompositeNode::resolveUsingNodes(const NodeMap & blocks)
    {
      NodePtrList::iterator nodeItr(nodes_->begin()),e(nodes_->end());
      for(;nodeItr!=e;++nodeItr)
      {
        if((**nodeItr).type() == "using")
        {
          // find the block
          string blockName = (**nodeItr).name;
          NodeMap::const_iterator blockPtrItr = blocks.find(blockName);
          if(blockPtrItr == blocks.end()) {
             throw edm::Exception(errors::Configuration,"")
               << "Cannot find parameter block " << blockName;
          }

          // insert each node in the UsingBlock into the list
          CompositeNode * psetNode = dynamic_cast<CompositeNode *>(blockPtrItr->second.get());
          assert(psetNode != 0);
          NodePtrListPtr params = psetNode->nodes();


          //@@ is it safe to delete the UsingNode now?
          nodes_->erase(nodeItr);

          for(NodePtrList::const_iterator paramItr = params->begin();
              paramItr != params->end(); ++paramItr)
          {
            // Using blocks get inserted at the beginning, just for convenience
            // Make a copy of the node, so it can be modified
            nodes_->push_front( NodePtr((**paramItr).clone()) );
          }

          // better start over, since list chnged,
          // just to be safe
          nodeItr = nodes_->begin();
        }

        else
        {
          // if it isn't a using node, check the grandchildren
          (**nodeItr).resolveUsingNodes(blocks);
        }
      } // loop over subnodes
    }


    void CompositeNode::insertInto(ParameterSet & pset) const
    {
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(;i!=e;++i)
      {
        (**i).insertInto(pset);
      }
    }

  }
}

