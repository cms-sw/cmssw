#ifndef ParameterSet_ParseResultsTweaker_h
#define ParameterSet_ParseResultsTweaker_h

/** Applies preprocessing to a ParseResults object.
    Designed to implement nodes for modifications
    such as copy, modify, and rename

    \Author Rick Wilkinson

  */

#include "FWCore/ParameterSet/interface/parse.h"
#include <map>
#include <string>

namespace edm {
  namespace pset {

    class ParseResultsTweaker 
    {
    public:
      ParseResultsTweaker() {}

      void process(ParseResults & parseResults);

    private:
      void clear();

      /// pulls the names out and remembers the Nodes
      void sortNodes(const NodePtrListPtr & parseResults);

      /// Only looks in top-level modules and sources
      /// inlines all parameters in the block
      /// and erases the UsingNode itself
      void processUsingBlocks();
      void processUsingBlock(NodePtrList::iterator & usingNodeItr, 
                             ModuleNode * moduleNode);
      
      void processCopyNode(const NodePtr & n);

      void processRenameNode(const NodePtr & n);

      void processReplaceNode(const NodePtr & n);


      /// parameters are specified by dot-delimited names.
      /// this method walks the tree 
      NodePtr findInPath(const std::string & path);

      /// throws a ConfigurationError if not found
      NodePtr findModulePtr(const std::string & name);
  
      /// calls findModulePtr, and does the cast
      ModuleNode * findModule(const std::string & name);

      /// puts the parts back together to return the ParseResults
      void reassemble(NodePtrListPtr & parseResults);


      /// for sorting NodePtrs by name
      typedef std::map<std::string, NodePtr> NodePtrMap;

      /// Nodes which represent top-level PSet blocks
      NodePtrMap blocks_;

      /// Nodes which copy modules and sources
      NodePtrList copyNodes_;

      /// Nodes which rename modules and sources
      NodePtrList renameNodes_;

      /// Nodes which replace parameters, sets, or modules
      NodePtrList replaceNodes_;

      /// key is the name of the module or source
      NodePtrMap modulesAndSources_;

      /// Everything but modules, sources, & modification nodes
      NodePtrList everythingElse_;
    };
  }
}

#endif

