#include "FWCore/ParameterSet/interface/CompositeNode.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <ostream>

namespace edm {

  namespace pset {

    CompositeNode::CompositeNode(const CompositeNode & n)
    : Node(n),
      nodes_(new NodePtrList)
    {
      NodePtrList::const_iterator i(n.nodes_->begin()),e(n.nodes_->end());
      for(; i != e; ++i)
      {
         nodes_->push_back( NodePtr((**i).clone()) );
      }
    }


    void CompositeNode::acceptForChildren(Visitor& v) const
    {
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(; i != e; ++i)
        {
          (*i)->accept(v);
        }
    }


    void CompositeNode::print(std::ostream& ost, Node::PrintOptions options) const
    {
      ost << "{\n";
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(; i != e; ++i)
        {
          (**i).print(ost, options);
          ost << "\n";
        }

      ost << "}\n";
    }


    void CompositeNode::locate(const std::string & s, std::ostream & out) const
    {
      // first check this name
      Node::locate(s, out);
      // now all the subnodes
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(; i != e; ++i)
      {
        (**i).locate(s, out);
      }
    }


    void CompositeNode::setModified(bool value) 
    {
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(; i != e; ++i)
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


    void CompositeNode::setCloned(bool value)
    {
      // set this node itself
      Node::setCloned(value);
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(; i != e; ++i)
      {
        (*i)->setCloned(value);
      }
    }


    void CompositeNode::setAsChildrensParent()
    {
      NodePtrList::iterator i(nodes_->begin()),e(nodes_->end());
      for(; i != e; ++i)
      {
        (**i).setParent(this);
        // make child register with grandchildren
        (**i).setAsChildrensParent();
      }
    }


    NodePtrListPtr CompositeNode::children() const
    {
      NodePtrListPtr result(new NodePtrList);
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(; i != e; ++i)
      {
        if((**i).isInclude())
        {
          CompositeNode * includeNode = dynamic_cast<CompositeNode *>(i->get());
          NodePtrListPtr includedChildren = includeNode->children();
          result->insert(result->end(), includedChildren->begin(), includedChildren->end());
        }
        else 
        {
          result->push_back(*i);
        }
      }
      return result;
    } 


    bool CompositeNode::findChild(const std::string & child, NodePtr & result)
    {
      NodePtrListPtr kids = children();
      NodePtrList::const_iterator i(kids->begin()),e(kids->end());
      for(; i != e; ++i)
      {
         if((*i)->name() == child) {
           result = *i;
           return true;
         }
      }

      // uh oh.  Didn't find it.
      return false;
    }


    void CompositeNode::removeChild(const std::string & child) 
    {
      NodePtrList::iterator i(nodes_->begin()),e(nodes_->end());
      for(; i != e; ++i)
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
      for(; i != e; ++i)
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
        << "Cannot find node " << child->name() << " to erase in " << name();
      
  }
    

    void CompositeNode::resolve(std::list<std::string> & openFiles,
                                std::list<std::string> & sameLevelIncludes,
                                bool strict)
    {
      // make sure that no siblings are IncludeNodes with the 
      // same name
      // IncludeNodes are transparent, so they don't get a new branch on
      // the family tree.  If it's another CompositeNode, it gets its
      // own stack
      std::list<std::string> newList;
      std::list<std::string> & thisLevelIncludes = 
                             isInclude()
                             ? sameLevelIncludes
                             : newList; 

      // make a copy, in case a node deletes itself
      NodePtrList copyOfNodes = *nodes_;
      NodePtrList::const_iterator i(copyOfNodes.begin()),e(copyOfNodes.end());
      for(; i != e; ++i)
      {
        (**i).resolve(openFiles, thisLevelIncludes, strict);
      }
    }


    void CompositeNode::resolveUsingNodes(const NodeMap & blocks, bool strict)
    {
      NodePtrList::iterator nodeItr(nodes_->begin()),e(nodes_->end());
      for(; nodeItr != e; ++nodeItr)
      {
        if((**nodeItr).type() == "using")
        {
          /*
          // using nodes in .cfis are either deprecated or forbidden
          if((**nodeItr).includeParentSuffix() == "cfi")
          {
             std::string message = "using statements should not be used in .cfi files.\nfrom:";
             message += (**nodeItr).traceback();
             if(strict)
             {
               throw edm::Exception(errors::Configuration,"") << message;
             }
             else
             {
                edm::LogWarning("Configuration") << message;
             }
          }
          */

          // find the block
          std::string blockName = (**nodeItr).name();
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

          for(NodePtrList::const_iterator paramItr = params->begin(), paramItrEnd = params->end();
              paramItr != paramItrEnd; ++paramItr)
          {
            // Using blocks get inserted at the beginning, just for convenience
            // Make a copy of the node, so it can be modified
            NodePtr newNode((**paramItr).clone());
            // if this node is cloned, propagate that to the children
            newNode->setCloned(isCloned());
            nodes_->push_front( newNode );
          }

          // better start over, since list chnged,
          // just to be safe
          nodeItr = nodes_->begin();
        }

        else
        {
          // if it isn't a using node, check the grandchildren
          (**nodeItr).resolveUsingNodes(blocks, strict);
        }
      } // loop over subnodes
    }


    void CompositeNode::insertInto(ParameterSet & pset) const
    {
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(; i != e; ++i)
      {
        (**i).insertInto(pset);
      }
    }


    void CompositeNode::validate() const
    {
      // make sure no node is duplicated
      std::vector<std::string> nodeNames;
      nodeNames.reserve(nodes_->size());
      NodePtrListPtr kids = children();
      NodePtrList::const_iterator i(kids->begin()),e(kids->end());
      for(; i != e; ++i)
      {
        // let unnamed node go, because they might be in the process
        if((**i).name() != "" && (**i).name() != "nameless" 
           && (**i).type() != "includeRenamed" && (**i).type() != "es_prefer")
        {
          if(std::find(nodeNames.begin(), nodeNames.end(), (**i).name())
             == nodeNames.end())
          {
            nodeNames.push_back((**i).name());
            // have the child validate itself
            (**i).validate();
          }
          else
          {
            throw edm::Exception(errors::Configuration,"") 
              << "Duplicate node name: parameter " << (**i).name() << " in " << name()
              << "\nfrom: " << traceback();
          }
        }
      }
 
    }

  }
}

