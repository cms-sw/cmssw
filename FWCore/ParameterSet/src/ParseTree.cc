#include "FWCore/ParameterSet/interface/ParseTree.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ModuleNode.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/ParameterSet/interface/IncludeNode.h"
#include "FWCore/ParameterSet/interface/PSetNode.h"
#include "FWCore/ParameterSet/interface/Nodes.h"
#include "FWCore/ParameterSet/interface/parse.h"

using std::string;
using std::vector;
using std::map;

#include <iostream>
#include<iterator>

namespace edm {
  namespace pset {

    ParseTree::ParseTree(const string & configString)
    : blocks_(),
      blockCopyNodes_(),
      blockRenameNodes_(),
      blockReplaceNodes_(),
      copyNodes_(),
      renameNodes_(),
      replaceNodes_(),
      modulesAndSources_(),
      nodes_(parse(configString.c_str()))
    {
      process();
    }


    PSetNode * ParseTree::getProcessNode() const
    {
      NodePtr processPSetNodePtr = nodes_->front();
      edm::pset::PSetNode * processPSetNode
        = dynamic_cast<edm::pset::PSetNode*>(processPSetNodePtr.get());
      assert(processPSetNode != 0);
      return processPSetNode;
    }


    CompositeNode * ParseTree::top() const
    {
      NodePtr nodePtr = nodes_->front();
      edm::pset::CompositeNode * node
        = dynamic_cast<edm::pset::CompositeNode*>(nodePtr.get());
      assert(node != 0);
      return node;
    }



    std::vector<std::string> ParseTree::modules() const
    {
      std::vector<std::string> result;
      result.reserve(modulesAndSources_.size());
      for(NodePtrMap::const_iterator moduleMapItr = modulesAndSources_.begin();
          moduleMapItr != modulesAndSources_.end(); ++moduleMapItr)
      {
        result.push_back(moduleMapItr->first);
      }
      return result;
    }


    std::vector<std::string> ParseTree::modulesOfType(const std::string & s) const
    {
      std::vector<std::string> result;
      for(NodePtrMap::const_iterator moduleMapItr = modulesAndSources_.begin();
          moduleMapItr != modulesAndSources_.end(); ++moduleMapItr)
      {
        if(moduleMapItr->second->type() == s)
        {
          result.push_back(moduleMapItr->first);
        }
      }
      return result;
    }



    void ParseTree::process()
    {
      clear();
      
      CompositeNode * topLevelNode = top();

      // make whatever backwards link you can now.  Include Nodes
      // can add more as needed
      topLevelNode->setAsChildrensParent();
      // find any include nodes
      // maybe someday list the current file as an open file,
      // so it never gets circularly included
      std::list<std::string> openFiles;
      std::list<std::string> sameLevelIncludes;
      topLevelNode->resolve(openFiles, sameLevelIncludes);
      // make the final backwards linksa.  Needed?
      //processNode->setAsChildrensParent();

      NodePtrListPtr contents = topLevelNode->nodes();
      sortNodes(contents);

      // maybe we don't have to do anything
      if( !copyNodes_.empty() || !replaceNodes_.empty() 
       || !renameNodes_.empty() || !blocks_.empty() )
      {
        // pull out the operations on shared blocks, and do them.
        findBlockModifiers(copyNodes_, blockCopyNodes_);
        findBlockModifiers(renameNodes_, blockRenameNodes_);
        findBlockModifiers(replaceNodes_, blockReplaceNodes_);
        
        NodePtrList::iterator nodeItr;
        // do copies
        for(nodeItr = blockCopyNodes_.begin();
            nodeItr != blockCopyNodes_.end(); ++nodeItr)
        {
          processCopyNode(*nodeItr, blocks_);
        }

        // do renames before replaces
        for(nodeItr = blockRenameNodes_.begin();
            nodeItr != blockRenameNodes_.end(); ++nodeItr)
        {
          processRenameNode(*nodeItr, blocks_);
        }

        // now replace nodes
        for(nodeItr = blockReplaceNodes_.begin();
            nodeItr != blockReplaceNodes_.end(); ++nodeItr)
        {
          processReplaceNode(*nodeItr, blocks_);
        }

        // NOTE: We only bother inlining the Using blocks
        // if there's a chance the parameters will be modified.
        // If not, they'll get done later.
        processUsingBlocks();

        // do copies
        for(nodeItr = copyNodes_.begin();
            nodeItr != copyNodes_.end(); ++nodeItr)
        {
          processCopyNode(*nodeItr, modulesAndSources_);
        }

        // do renames before replaces
        for(nodeItr = renameNodes_.begin();
            nodeItr != renameNodes_.end(); ++nodeItr)
        {
          processRenameNode(*nodeItr, modulesAndSources_);
        }

        // now replace nodes
        for(nodeItr = replaceNodes_.begin();
            nodeItr != replaceNodes_.end(); ++nodeItr) 
        {
          processReplaceNode(*nodeItr, modulesAndSources_);
        }

        // reassemble(contents);
      }
    }


    void ParseTree::replace(const std::string & dotDelimitedPath,
                            const std::string & value)
    {
      NodePtr entryNode(new EntryNode("replace", dotDelimitedPath, value, false));
      NodePtr replaceNode(new ReplaceNode("replace", dotDelimitedPath, entryNode, true, -1));
      top()->nodes()->push_back(replaceNode);
    }


    void ParseTree::replace(const std::string & dotDelimitedPath,
                            const std::vector<std::string> & values)
    {
      StringListPtr strings(new StringList(values));
      NodePtr vEntryNode(new VEntryNode("replace", dotDelimitedPath, strings, false));
      NodePtr replaceNode(new ReplaceNode("replace", dotDelimitedPath, vEntryNode, true, -1));
      top()->nodes()->push_back(replaceNode);
    }


    void ParseTree::print(const std::string & dotDelimitedNode) const
    {
      NodePtr nodePtr = findInPath(dotDelimitedNode);
      nodePtr->print(std::cout, Node::EXPANDED);
    }

    std::string ParseTree::typeOf(const std::string & dotDelimitedNode) const
    {
      return findInPath(dotDelimitedNode)->type();
    }


    std::string ParseTree::value(const std::string & dotDelimitedNode) const
    {
      string result = "";
      NodePtr nodePtr = findInPath(dotDelimitedNode);
      EntryNode * entryNode = dynamic_cast<EntryNode *>(nodePtr.get());
      if(entryNode == 0)
      {
        throw edm::Exception(errors::Configuration,"")
        << dotDelimitedNode << " is not a single entry";
      }
      else
      {
        result = entryNode->value();
      }
      return result;
    }


      /// only works for VEntryNodes
    std::vector<std::string> ParseTree::values(const std::string & dotDelimitedNode) const
    {
      std::vector<std::string> result;
      NodePtr nodePtr = findInPath(dotDelimitedNode);
      VEntryNode * vEntryNode = dynamic_cast<VEntryNode *>(nodePtr.get());
       
      if(vEntryNode == 0)
      {
        throw edm::Exception(errors::Configuration,"")
        << dotDelimitedNode << " is not a vector of values";
      }
      else 
      {
        result = *(vEntryNode->value());
      }
      return result;
    }

    // names of the nodes below this one.  Includes are transparent

    std::vector<std::string> ParseTree::children(const std::string & dotDelimitedNode) const
    {
      std::vector<std::string> result; 
      //TODO
      return result;
    }



    void ParseTree::clear() 
    {
      blocks_.clear();
      copyNodes_.clear();
      renameNodes_.clear();
      replaceNodes_.clear();
      blockCopyNodes_.clear();
      blockRenameNodes_.clear();
      blockReplaceNodes_.clear();
      modulesAndSources_.clear();
    }


    void ParseTree::sortNodes(const NodePtrListPtr & nodes)
    {

      NodePtrList topLevelNodes;
      findTopLevelNodes(*nodes, topLevelNodes);

      for(NodePtrList::const_iterator nodeItr = topLevelNodes.begin();
          nodeItr != topLevelNodes.end(); ++nodeItr)
      {
        // see what the type is
        string type = (*nodeItr)->type();
        string name = (*nodeItr)->name();
        // see if it's ont of the many types of ModuleNode first
        ModuleNode * moduleNode = dynamic_cast<ModuleNode *>((*nodeItr).get());
        if(moduleNode != 0) 
        {
          //@@TODO FIX HACK! unnamed es_prefers need to be unmodifiable for the
          // time being, since they can have the same class as a different es_source
          if(type != "es_prefer") 
          {
            // unnamed modules are named after class
            if(name == "nameless" || name == "" || name=="main_es_input") 
            {
              name = moduleNode->className();
            }

            // double-check that no duplication
            NodePtrMap::iterator moduleMapItr = modulesAndSources_.find(name);
            if(moduleMapItr != modulesAndSources_.end()) 
            {
              std::ostringstream firstTrace, secondTrace;
              moduleNode->printTrace(secondTrace);
              moduleMapItr->second->printTrace(firstTrace);
              if(firstTrace.str().empty()) firstTrace << "main config\n";
              if(secondTrace.str().empty()) secondTrace << "main config\n";
              throw edm::Exception(errors::Configuration,"") 
               << "Duplicate definition of " << name
               << "\nfirst: " << firstTrace.str()
               << "second: " << secondTrace.str()
               << "Please edit the configuration so it is only defined once";
              //edm::LogWarning("ParseTree") << "Duplicate definition of "
              //<< name << ". Only last one will be kept.";
            }
            modulesAndSources_[name] = *nodeItr;
          }
        } // moduleNode
  
        else if(type == "block" || type == "PSet") {
          blocks_[name] = *nodeItr;
        }

        else if(string(type,0,7) == "replace") {
          replaceNodes_.push_back(*nodeItr);
        }

        else if(type == "copy") {
          copyNodes_.push_back(*nodeItr);
        }

        else if(type == "rename") {
          renameNodes_.push_back(*nodeItr);
        }

      }
    }


    void ParseTree::processUsingBlocks()
    {
      // look for blocks-within-blocks first
      for(NodePtrMap::iterator blockItr = blocks_.begin();
          blockItr != blocks_.end(); ++blockItr)
      {
        blockItr->second->resolveUsingNodes(blocks_);
      }

      for(NodePtrMap::iterator moduleItr = modulesAndSources_.begin();
          moduleItr != modulesAndSources_.end();  ++moduleItr)
      {
        moduleItr->second->resolveUsingNodes(blocks_);
      }

      // maybe there's a using statement inside a replace PSet?
      // You never know.
      for(NodePtrList::iterator replaceItr = replaceNodes_.begin();
          replaceItr != replaceNodes_.end();  ++replaceItr)
      {
        ReplaceNode * replaceNode = dynamic_cast<ReplaceNode *>(replaceItr->get());
        CompositeNode * compositeNode = dynamic_cast<CompositeNode *>(replaceNode->value().get());
        if(compositeNode != 0)
        {
          compositeNode->resolveUsingNodes(blocks_);
        }
      } 

    }


    void ParseTree::processCopyNode(const NodePtr & n,
                                ParseTree::NodePtrMap  & targetMap)
    {
      assert(false);
     /*
      const CopyNode * copyNode = dynamic_cast<const CopyNode*>(n.get());
      assert(copyNode != 0);

      NodePtr fromPtr = findPtr(copyNode->from(), targetMap);
      NodePtr toPtr(fromPtr->clone());
      toPtr->setName(copyNode->to());

      // and add it in the maps here
      targetMap[copyNode->to()] = toPtr;
      removeNode(n);
      */
    }


    void ParseTree::processRenameNode(const NodePtr & n,
                                  ParseTree::NodePtrMap  & targetMap)
    {
      const RenameNode * renameNode = dynamic_cast<const RenameNode*>(n.get());
      assert(renameNode != 0);

      NodePtr targetPtr = findPtr(renameNode->from(), targetMap);
      targetPtr->setName(renameNode->to());

      // and replace it in the maps here
      targetMap[renameNode->to()] = targetPtr;
      targetMap.erase(renameNode->from());
      // get rid of the renameNode
      removeNode(n);
    }


    void ParseTree::processReplaceNode(NodePtr & n,
                                ParseTree::NodePtrMap  & targetMap)
    {
      NodePtr targetPtr = findInPath(n->name(), targetMap);
      ReplaceNode * replaceNode = dynamic_cast<ReplaceNode*>(n.get());
      assert(replaceNode != 0);
      // see if we need to resolve this replace node
      if(replaceNode->value()->type() == "dotdelimited")
      {
        NodePtr newValue( findInPath(replaceNode->value()->name(), blocks_) );
        replaceNode->setValue(newValue);
      }
      checkOkToModify(replaceNode, targetPtr);
      // we're here to replace it.  So replace it.
      targetPtr->replaceWith(replaceNode);
      removeNode(n);
    }


    void ParseTree::removeNode(const NodePtr & victim)
    {
      CompositeNode * parent  = dynamic_cast<CompositeNode *>(victim->getParent());
      assert(parent != 0);
      parent->removeChild(victim->name());
    }


    NodePtr ParseTree::findInPath(const std::string & path) const
    {
      // try blocks_, then modulesAndSources_
      NodePtr result;
      try 
      {
        result = findInPath(path, modulesAndSources_);
      }
      catch(const edm::Exception & e)
      {   
        // may throw... that's OK.
        result = findInPath(path, blocks_);
      }
      return result;
    }


    NodePtr ParseTree::findInPath(const string & path, 
                                  const ParseTree::NodePtrMap  & targetMap) const
    {
      typedef vector<string> stringvec_t;
      stringvec_t pathElements = tokenize(path, ".");
      stringvec_t::const_iterator it =  pathElements.begin();
      stringvec_t::const_iterator end = pathElements.end();

      // top level should be the module
      NodePtr currentPtr = findPtr(*it, targetMap);
      // dig deeper, if we have to
      ++it;
      while(it != end)
      {
        Node * currentNode = currentPtr.get();
        CompositeNode * compositeNode = dynamic_cast<CompositeNode*>(currentNode);
        if(compositeNode == 0)
        {
          throw edm::Exception(errors::Configuration,"No such element") 
             << "Not a composite node: " << currentNode->name() << " in " << path;
        }
        if(compositeNode->findChild(*it, currentPtr) == false)
        {
          throw edm::Exception(errors::Configuration,"No such element")
             << "Could not find: " << *it << " in " << currentNode->name();
        }


        ++it; 
      }
    
      return currentPtr;
    }


    NodePtr ParseTree::findPtr(const string & name, 
                               const ParseTree::NodePtrMap  & targetMap) const 
    {
      NodePtrMap::const_iterator mapItr = targetMap.find(name);
      if(mapItr == targetMap.end()) {
        throw edm::Exception(errors::Configuration,"No Such Object") 
                << "Cannot find " << name;
      }
      return mapItr->second;
    }


    void ParseTree::findBlockModifiers(NodePtrList & modifierNodes,
                                                 NodePtrList & blockModifiers)
    {
      // need to be careful not to invalidate iterators when we erase
      NodePtrList::iterator modifierItr = modifierNodes.begin();
      while(modifierItr != modifierNodes.end())
      {
        NodePtrList::iterator next = modifierItr;
        ++next;

        // see if this name is a block name
        string topLevel = tokenize((**modifierItr).name(), ".")[0];
        if(blocks_.find(topLevel) != blocks_.end())
        {
          blockModifiers.push_back(*modifierItr);
          modifierNodes.erase(modifierItr);
        }
        modifierItr = next;
      }
    }


    void ParseTree::findTopLevelNodes(const NodePtrList & input, NodePtrList & output)
    {
      for(NodePtrList::const_iterator inputNodeItr = input.begin();
          inputNodeItr  != input.end(); ++inputNodeItr)
      {
        // make IncludeNodes transparent
        if((**inputNodeItr).type().substr(0,7) == "include")
        {
          const IncludeNode * includeNode 
            = dynamic_cast<const IncludeNode*>(inputNodeItr->get());
          assert(includeNode != 0);
          // recursive call!
          findTopLevelNodes(*(includeNode->nodes()), output);
          // just to make sure recursion didn't bite me
          assert((**inputNodeItr).type().substr(0,7) == "include");
        }
        else 
        {
          output.push_back(*inputNodeItr);
        }
      }
    }


    void ParseTree::checkOkToModify(const ReplaceNode * replaceNode, NodePtr targetNode)
    {
      if(targetNode->isModified() && !(replaceNode->okToRemodify()))
      {
        throw edm::Exception(errors::Configuration)
          << "Cannot replace a node that has already been modified: " << targetNode->name();
      }
    }
      

          
  }  // pset namespace
} // edm namespace

