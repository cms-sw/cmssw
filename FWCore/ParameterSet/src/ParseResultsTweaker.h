#ifndef ParameterSet_ParseResultsTweaker_h
#define ParameterSet_ParseResultsTweaker_h

/** Applies preprocessing to a ParseResults object.
    Designed to implement nodes for modifications
    such as copy, modify, and rename

    \Author Rick Wilkinson

  */

#include "FWCore/ParameterSet/interface/Node.h"
#include <map>
#include <string>

namespace edm {
  namespace pset {

    class ParseResultsTweaker 
    {
    public:
      /// for sorting NodePtrs by name
      typedef std::map<std::string, NodePtr> NodePtrMap;

      ParseResultsTweaker() {}

      void process(ParseResults & parseResults);

    private:
      void clear();

      /// pulls the names out and remembers the Nodes
      void sortNodes(const NodePtrListPtr & parseResults);

      /// Only looks in top-level blocks, modules and sources
      /// inlines all parameters in the block
      /// and erases the UsingNode itself
      void processUsingBlocks();

      /// targetMap will ordinarily be one of the data members
      void processCopyNode(const NodePtr & n, NodePtrMap & targetMap);

      void processRenameNode(const NodePtr & n, NodePtrMap & targetMap);

      void processReplaceNode(const NodePtr & n, NodePtrMap & targetMap);

      /// once we're done with a rename/replace node, we throw it away
      void removeNode(const NodePtr & victim);

      /// parameters are specified by dot-delimited names.
      /// this method walks the tree 
      NodePtr findInPath(const std::string & path, NodePtrMap & nodeMap);

      /// throws a ConfigurationError if not found
      NodePtr findPtr(const std::string & name, NodePtrMap & nodeMap);
  
      /// utilities to modify a block before it's inlined

      /// pulls any modifier commands (rename, copy, replace, etc.)
      /// out of the list they're in if they refer to an existing block
      /// they're stored in blockModifiers and erased from the input.
      void findBlockModifiers(NodePtrList & modifierNodes, 
                              NodePtrList & blockModifiers);

      /// lists all the top-level nodes, while treating the IncludeNodes
      /// as transparent
      void findTopLevelNodes(const NodePtrList & input, NodePtrList & output);

      /// Nodes which represent top-level PSet blocks
      NodePtrMap blocks_;

      /// Nodes which modify blocks.  These are processed first
      NodePtrList blockCopyNodes_;
      NodePtrList blockRenameNodes_;
      NodePtrList blockReplaceNodes_;

      /// Nodes which copy modules and sources
      NodePtrList copyNodes_;

      /// Nodes which rename modules and sources
      NodePtrList renameNodes_;

      /// Nodes which replace parameters, sets, or modules
      NodePtrList replaceNodes_;

      /// key is the name of the module or source
      NodePtrMap modulesAndSources_;
    };
  }
}

#endif

