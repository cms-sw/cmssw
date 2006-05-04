#include "FWCore/ParameterSet/src/ParseResultsTweaker.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "boost/tokenizer.hpp"

using std::string;
using std::vector;
using std::map;
#include <iostream>
#include<iterator>

namespace edm {
  namespace pset {

    void ParseResultsTweaker::process(ParseResults & parseResults)
    {
      // find the node that represents the process
      PSetNode * processNode = 0;
      // assume it's the only one in a sane .cfg file
      if(parseResults->size() == 1) {
        processNode = dynamic_cast<PSetNode*>(parseResults->front().get());
      }

      // if it's not a simple config file, with one process node, bail out
      if(processNode == 0) {
        edm::LogWarning("ParseResultsTweaker") << "Cannot find process node";
      } else {
        // PSetNode -> ContentsNode -> NodePtrListPtr
        NodePtrListPtr contents = processNode->value_.nodes_;
        sortNodes(contents);

        // maybe we don't have to do anything
        if(!copyNodes_.empty() || !replaceNodes_.empty() || !renameNodes_.empty()) 
        {
          // pull out the operations on shared blocks, and do them.
          findBlockModifiers(copyNodes_, blockCopyNodes_);
          findBlockModifiers(renameNodes_, blockRenameNodes_);
          findBlockModifiers(replaceNodes_, blockReplaceNodes_);
          
          NodePtrList::const_iterator nodeItr;
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

          reassemble(contents);
        }
      }
    }



    void ParseResultsTweaker::clear() 
    {
      blocks_.clear();
      copyNodes_.clear();
      renameNodes_.clear();
      replaceNodes_.clear();
      blockCopyNodes_.clear();
      blockRenameNodes_.clear();
      blockReplaceNodes_.clear();
      modulesAndSources_.clear();
      everythingElse_.clear();
    }


    void ParseResultsTweaker::sortNodes(const NodePtrListPtr & nodes)
    {

      for(NodePtrList::const_iterator nodeItr = nodes->begin();
          nodeItr != nodes->end(); ++nodeItr)
      {
        // see what the type is
        string type = (*nodeItr)->type();
        string name = (*nodeItr)->name;

        if(type == "module" || type == "es_module"
        || type == "source" || type == "es_source") 
        {
          // unnamed modules are named after class
          if(name == "nameless" || name == "" || name=="main_es_input") 
          {
            ModuleNode * moduleNode = dynamic_cast<ModuleNode *>((*nodeItr).get());
            name = moduleNode->class_;
          }

          // double-check that no duplication
          NodePtrMap::iterator moduleMapItr = modulesAndSources_.find(name);
          if(moduleMapItr != modulesAndSources_.end()) {
            //throw edm::Exception(errors::Configuration,"") 
            // << "Duplicate definition of " << name << std::endl;
            edm::LogWarning("ParseResultsTweaker") << "Duplicate definition of "
            << name << ". Only last one will be kept.";
          }
          modulesAndSources_[name] = *nodeItr;
        }

        else if(type == "block") {
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

        else {
          everythingElse_.push_back(*nodeItr);
        }
      }
    }


    void ParseResultsTweaker::processUsingBlocks()
    {
      for(NodePtrMap::iterator moduleItr = modulesAndSources_.begin();
          moduleItr != modulesAndSources_.end();  ++moduleItr)
      {
        ModuleNode * moduleNode = dynamic_cast<ModuleNode *>(moduleItr->second.get()); 
        assert(moduleNode != 0);
        NodePtrListPtr nodes = moduleNode->nodes_;

        // look for a Using block in ithe top level of this module
        for(NodePtrList::iterator nodeItr = nodes->begin();
            nodeItr != nodes->end(); ++nodeItr)
        {
          if((*nodeItr)->type() == "using") {
            processUsingBlock(nodeItr, moduleNode);
            // better start over, since list chnged,
            // just to be safe
            nodeItr = nodes->begin();
          }
        }
      }  // loop over modules & sources
    }
    

    void ParseResultsTweaker::processUsingBlock(NodePtrList::iterator & usingNodeItr, 
                                                ModuleNode * moduleNode) 
    {
      // find the block
      string blockName = (*usingNodeItr)->name;
      NodePtrMap::const_iterator blockPtrItr = blocks_.find(blockName);
      if(blockPtrItr == blocks_.end()) {
         throw edm::Exception(errors::Configuration,"")
           << "Cannot find parameter block " << blockName;
      }
      
      // insert each node in the UsingBlock into the list
      PSetNode * psetNode = dynamic_cast<PSetNode *>(blockPtrItr->second.get());
      assert(psetNode != 0);
      NodePtrListPtr params = psetNode->value_.nodes();
      
      // find the contents of the Module
      NodePtrListPtr moduleContents = moduleNode->nodes();

      //@@ is it safe to delete the UsingNode now?
      moduleContents->erase(usingNodeItr);

      for(NodePtrList::const_iterator paramItr = params->begin();
          paramItr != params->end(); ++paramItr)
      {
        // Using blocks get inserted at the beginning, just for convenience
        // Make a copy of the node, so it can be modified
        moduleContents->push_front( NodePtr((**paramItr).clone()) );
      } 
    }


    void ParseResultsTweaker::processCopyNode(const NodePtr & n,
                                ParseResultsTweaker::NodePtrMap  & targetMap)
    {
      const CopyNode * copyNode = dynamic_cast<const CopyNode*>(n.get());
      assert(copyNode != 0);

      NodePtr fromPtr = findPtr(copyNode->from(), targetMap);
      NodePtr toPtr(fromPtr->clone());
      toPtr->name = copyNode->to();

      // and add it in the maps here
      targetMap[copyNode->to()] = toPtr;
    }


    void ParseResultsTweaker::processRenameNode(const NodePtr & n,
                                  ParseResultsTweaker::NodePtrMap  & targetMap)
    {
      const RenameNode * renameNode = dynamic_cast<const RenameNode*>(n.get());
      assert(renameNode != 0);

      NodePtr targetPtr = findPtr(renameNode->from(), targetMap);
      targetPtr->name = renameNode->to();

      // and replace it in the maps here
      targetMap[renameNode->to()] = targetPtr;
      targetMap.erase(renameNode->from());
    }


    void ParseResultsTweaker::processReplaceNode(const NodePtr & n,
                                ParseResultsTweaker::NodePtrMap  & targetMap)
    {
      NodePtr targetPtr = findInPath(n->name, targetMap);
      const ReplaceNode * replaceNode = dynamic_cast<const ReplaceNode*>(n.get());
      assert(replaceNode != 0);
      // we're here to replace it.  So replace it.
      targetPtr->replaceWith(replaceNode);
    }


    std::vector<std::string> ParseResultsTweaker::parsePath(const std::string & path)
    {
      typedef boost::char_separator<char>   separator_t;
      typedef boost::tokenizer<separator_t> tokenizer_t;

      vector<string> pathElements;
      separator_t  sep("."); // separator for elements in path
      tokenizer_t  tokens(path, sep);
      std::copy(tokens.begin(),
                tokens.end(),
                std::back_inserter<vector<string> >(pathElements));
      return pathElements;
    }


    NodePtr ParseResultsTweaker::findInPath(const string & path, 
                                            ParseResultsTweaker::NodePtrMap  & targetMap)
    {
      typedef vector<string> stringvec_t;
      stringvec_t pathElements = parsePath(path);
      stringvec_t::const_iterator it =  pathElements.begin();
      stringvec_t::const_iterator end = pathElements.end();

      // top level should be the module
      NodePtr currentPtr = findPtr(*it, targetMap);
      Node * currentNode = currentPtr.get(); 
      // dig deeper, if we have to
      ++it;
      while(it != end)
      {
        //  if this is a PSetNode, move down to the Contents
        PSetNode * psetNode = dynamic_cast<PSetNode*>(currentPtr.get());
        if(psetNode != 0) {
          currentNode = &(psetNode->value_);
        }

        CompositeNode * compositeNode = dynamic_cast<CompositeNode*>(currentNode);
        if(compositeNode == 0)
        {
          throw edm::Exception(errors::Configuration,"No such element") 
             << "Not a composite node: " << currentNode->name << " in " << path;
        }
        currentPtr = compositeNode->findChild(*it);

        ++it; 
      }
    
      return currentPtr;
    }


    NodePtr ParseResultsTweaker::findPtr(const string & name, 
                                         ParseResultsTweaker::NodePtrMap  & targetMap) 
    {
      NodePtrMap::iterator mapItr = targetMap.find(name);
      if(mapItr == targetMap.end()) {
        throw edm::Exception(errors::Configuration,"No Such Object") 
                << "Cannot find " << name;
      }
      return mapItr->second;
    }


    void ParseResultsTweaker::reassemble(NodePtrListPtr & contents)
    {
      contents->clear();
 
      // blocks go first
      for(NodePtrMap::const_iterator blockItr = blocks_.begin();
          blockItr != blocks_.end(); ++blockItr)
      {
        contents->push_back(blockItr->second);
      }

      // put on the (modified) modules
      for(NodePtrMap::const_iterator moduleItr = modulesAndSources_.begin(); 
          moduleItr != modulesAndSources_.end(); ++moduleItr)
      {
        contents->push_back(moduleItr->second);
      }

      // and everything else that was in the original.  Order shouldn't matter
      for(NodePtrList::const_iterator nodeItr = everythingElse_.begin();
          nodeItr != everythingElse_.end(); ++nodeItr)
      {
        contents->push_back(*nodeItr);
      }
    }


    void ParseResultsTweaker::findBlockModifiers(NodePtrList & modifierNodes,
                                                 NodePtrList & blockModifiers)
    {
      // need to be careful not to invalidate iterators when we erase
      NodePtrList::iterator modifierItr = modifierNodes.begin();
      while(modifierItr != modifierNodes.end())
      {
        NodePtrList::iterator next = modifierItr;
        ++next;

        // see if this name is a block name
        string topLevel = parsePath( (**modifierItr).name )[0];
        if(blocks_.find(topLevel) != blocks_.end())
        {
          blockModifiers.push_back(*modifierItr);
          modifierNodes.erase(modifierItr);
        }
        modifierItr = next;
      }
    }
          
  }  // pset namespace
} // edm namespace

