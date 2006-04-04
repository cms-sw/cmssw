#include "FWCore/ParameterSet/src/ParseResultsTweaker.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "boost/tokenizer.hpp"

using std::string;
using std::map;
#include <iostream>

namespace edm {
  namespace pset {

    void ParseResultsTweaker::process(ParseResults & parseResults)
    {
      clear();
      sortNodes(parseResults);

      // maybe we don't have to do anything
      if(!replaceNodes_.empty()) 
      {
  
        // NOTE: We only bother inlining the Using blocks
        // if there's a chance the parameters will be modified.
        // If not, they'll get done later.
        processUsingBlocks();

        NodePtrList::const_iterator nodeItr;

        // now replace nodes
        for(nodeItr = replaceNodes_.begin();
            nodeItr != replaceNodes_.end(); ++nodeItr) 
        {
          processReplaceNode(*nodeItr);
        }

        reassemble(parseResults);
      }
    }



    void ParseResultsTweaker::clear() 
    {
      replaceNodes_.clear();
      modulesAndSources_.clear();
      everythingElse_.clear();
    }


    void ParseResultsTweaker::sortNodes(ParseResults & parseResults)
    {
      NodePtrListPtr nodes = getContents(parseResults); 

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
          if(name == "nameless" || name == "") {
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

        else if(type == "block") {
          blocks_[name] = *nodeItr;
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
      NodePtrListPtr params = psetNode->value_.nodes_;
      
      // find the contents of the Module
      NodePtrListPtr moduleContents = moduleNode->nodes_;

      //@@ is it safe to delete the UsingNode now?
      moduleContents->erase(usingNodeItr);

      for(NodePtrList::const_iterator paramItr = params->begin();
          paramItr != params->end(); ++paramItr)
      {
        // Using blocks get inserted at the beginning, just for convenience
        // Might affect overwriting
        moduleNode->nodes_->push_front(*paramItr);
      } 
    }


    void ParseResultsTweaker::processReplaceNode(const NodePtr & n)
    {
      NodePtr targetPtr = findInPath(n->name);
      const ReplaceNode * replaceNode = dynamic_cast<const ReplaceNode*>(n.get());
      assert(replaceNode != 0);
      // we're here to replace it.  So replace it.
      targetPtr->replaceWith(replaceNode);
    }


    NodePtr ParseResultsTweaker::findInPath(const string & path)
    {
      typedef boost::char_separator<char>   separator_t;
      typedef boost::tokenizer<separator_t> tokenizer_t;

      separator_t  sep("."); // separator for elements in path
      tokenizer_t  tokens(path, sep);
      typedef std::vector<std::string> stringvec_t;
      stringvec_t  pathElements;
      std::copy(tokens.begin(),
                tokens.end(),
                std::back_inserter<stringvec_t>(pathElements));
       
      stringvec_t::const_iterator it =  pathElements.begin();
      stringvec_t::const_iterator end = pathElements.end();

      // top level should be the module
      NodePtr currentPtr = findModulePtr(*it);
      Node * currentNode = currentPtr.get(); 
      // dig deeper, if we have to
      ++it;
      while(it != end)
      {
        // if we're still digging, this must be a Composite node
        CompositeNode * compositeNode = dynamic_cast<CompositeNode*>(currentNode);
        if(currentNode == 0)
        {
          throw edm::Exception(errors::Configuration,"No such element") 
             << "Not a composite node: " << *it << " in " << path;
        }
        
        currentPtr = compositeNode->findChild(*it);

        // increment the iterator, and see if we're done
        if(++it != end)
        {
          // better be a PSet
          PSetNode * psetNode = dynamic_cast<PSetNode*>(currentPtr.get());
          if(psetNode == 0) {
            throw edm::Exception(errors::Configuration,"No such element")
             << "Not a PSet node: " << currentPtr->name << " in " << path;
          }
          currentNode = &(psetNode->value_);
        }
          
      }
    
      return currentPtr;
    }


    NodePtr ParseResultsTweaker::findModulePtr(const string & name) 
    {
      NodePtrMap::iterator moduleMapItr = modulesAndSources_.find(name);
      if(moduleMapItr == modulesAndSources_.end()) {
        throw edm::Exception(errors::Configuration,"No Such Module ") << "Cannot find " << name;
      }
      return moduleMapItr->second;
    }


    ModuleNode * ParseResultsTweaker::findModule(const string & name) 
    {
      NodePtr nodePtr = findModulePtr(name);
      ModuleNode * moduleNode = dynamic_cast<ModuleNode *>(nodePtr.get());
      return moduleNode;
    }


    void ParseResultsTweaker::reassemble(ParseResults & results)
    {
      NodePtrListPtr contents = getContents(results);
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



    NodePtrListPtr getContents(ParseResults & parseResults)
    {
      assert(parseResults->size() == 1);
      PSetNode * processNode = dynamic_cast<PSetNode*>(parseResults->front().get());
      if(processNode == 0) {
        throw edm::Exception(errors::Configuration,"") << "Top level is not a process";
      }
      // PSetNode -> ContentsNode -> NodePtrListPtr
      return processNode->value_.nodes_;
    }

    
  }  // pset namespace
} // edm namespace

