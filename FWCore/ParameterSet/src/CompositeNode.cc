#include "FWCore/ParameterSet/interface/CompositeNode.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
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


    void CompositeNode::print(ostream& ost, Node::PrintOptions options) const
    {
      ost << "{\n";
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(;i!=e;++i)
        {
          (**i).print(ost, options);
          ost << "\n";
        }

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
      bool result = Node::isModified();
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(;(i!=e) &&  !result;++i)
      {
        result = result || (*i)->isModified();
      }
      return result;
    }


    void CompositeNode::setAsChildrensParent()
    {
      NodePtrList::iterator i(nodes_->begin()),e(nodes_->end());
      for(;i!=e;++i)
      {
        (**i).setParent(this);
        // make child register with grandchildren
        (**i).setAsChildrensParent();
      }
    }


    bool CompositeNode::findChild(const string & child, NodePtr & result)
    {
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(;i!=e;++i)
      {
         if((*i)->name() == child) {
           result = *i;
           return true;
         }

         // IncludeNodes are transparent to this
         if((**i).type() == "include") 
         {
           CompositeNode * includeNode = dynamic_cast<CompositeNode *>((*i).get());
           assert(includeNode != 0);
           if(includeNode->findChild(child, result))
           {
             return true;
           }
         }
      }

      // uh oh.  Didn't find it.
      return false;
    }


    void CompositeNode::removeChild(const std::string & child) 
    {
      NodePtrList::iterator i(nodes_->begin()),e(nodes_->end());
      for(;i!=e;++i)
      {
        if((**i).name() == child)
        {
          nodes_->erase(i);
          // only set this node's modified flag
          Node::setModified(true);
          return;
        }
      }
 
      // if we didn't find it
      throw edm::Exception(errors::Configuration,"")
        << "Cannot find node " <<child << " to erase in " << name();

    }

    void CompositeNode::removeChild(const Node* child) 
  {
      NodePtrList::iterator i(nodes_->begin()),e(nodes_->end());
      for(;i!=e;++i)
      {
        if( &(**i) == child)
        {
          nodes_->erase(i);
          // only set this node's modified flag
          Node::setModified(true);
          return;
        }
      }
      
      // if we didn't find it
      throw edm::Exception(errors::Configuration,"")
        << "Cannot find node " <<child->name() << " to erase in " << name();
      
  }
    

    void CompositeNode::resolve(std::list<std::string> & openFiles,
                                std::list<std::string> & sameLevelIncludes)
    {
      // make sure that no siblings are IncludeNodes with the 
      // same name
      // IncludeNodes are transparent, so they don't get a new branch on
      // the family tree.  If it's another CompositeNode, it gets its
      // own stack
      std::list<std::string> newList;
      std::list<std::string> & thisLevelIncludes = (type() == "include")
                             ? sameLevelIncludes
                             : newList; 

      // make a copy, in case a node deletes itself
      NodePtrList copyOfNodes = *nodes_;
      NodePtrList::const_iterator i(copyOfNodes.begin()),e(copyOfNodes.end());
      for(;i!=e;++i)
      {
        (**i).resolve(openFiles, thisLevelIncludes);
      }
    }


    void CompositeNode::resolveUsingNodes(const NodeMap & blocks)
    {
      NodePtrList::iterator nodeItr(nodes_->begin()),e(nodes_->end());
      for(;nodeItr!=e;++nodeItr)
      {
        if((**nodeItr).type() == "using")
        {
          // find the block
          string blockName = (**nodeItr).name();
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

