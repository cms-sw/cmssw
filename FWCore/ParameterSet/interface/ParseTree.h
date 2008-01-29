#ifndef ParameterSet_ParseTree_h
#define ParameterSet_ParseTree_h

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

    class CompositeNode;
    class PSetNode;

    class ParseTree 
    {
    public:
      /// for sorting NodePtrs by name
      typedef std::map<std::string, NodePtr> NodePtrMap;

      static void setStrictParsing(bool strict);
      static void doReplaces(bool doOrNotDo);

      explicit ParseTree(const std::string & configString);

      /// the top-level process PSetNode
      PSetNode * getProcessNode() const;

      /// the top-level node if this may be a fragment
      CompositeNode * top() const;

      /// the names of all the modules we see
      std::vector<std::string> modules() const;
      /// all modules of a given type: service, es_source, looper, etc
      std::vector<std::string> modulesOfType(const std::string & s) const;

      /// processes all includes, renames, etc.
      void process();

      /// replaces the value of an entry
      void replace(const std::string & dotDelimitedNode, 
                   const std::string & value);

      void replace(const std::string & dotDelimitedNode, 
                   const std::vector<std::string> & values);
      
      void print(const std::string & dotDelimitedNode) const;

      std::string typeOf(const std::string & dotDelimitedNode) const;

      /// only works for EntryNodes inside modules.  Hope to include top-level PSets soon
      std::string value(const std::string & dotDelimitedNode) const;

      /// only works for VEntryNodes
      std::vector<std::string> values(const std::string & dotDelimitedNode) const;

      /// names of the nodes below this one.  Includes are transparent
      std::vector<std::string> children(const std::string & dotDelimitedNode) const;

      /// makes sure there are no duplicate names
      /// in modulesAndSources      
      void validate() const;

    private:
      /// doesn't change the nodes_ field.
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

      void processReplaceNode(NodePtr & n, NodePtrMap & targetMap);

      /// once we're done with a rename/replace node, we throw it away
      void removeNode(const NodePtr & victim);

      /// finds a node in a dot-delimited path in some different maps
      NodePtr findInPath(const std::string & path) const;
      /// parameters are specified by dot-delimited names.
      /// this method walks the tree 
      NodePtr findInPath(const std::string & path, const NodePtrMap & nodeMap) const;

      /// throws a ConfigurationError if not found
      NodePtr findPtr(const std::string & name, const NodePtrMap & nodeMap) const;
  
      /// utilities to modify a block before it's inlined

      /// pulls any modifier commands (rename, copy, replace, etc.)
      /// out of the list they're in if they refer to an existing block
      /// they're stored in blockModifiers and erased from the input.
      void findBlockModifiers(NodePtrList & modifierNodes, 
                              NodePtrList & blockModifiers);

      /// lists all the top-level nodes, while treating the IncludeNodes
      /// as transparent
      void findTopLevelNodes(const NodePtrList & input, NodePtrList & output);

      /// OK if ReplaceNode says it's OK to remodify, or if not already modify
      /// throws if not
      void checkOkToModify(const ReplaceNode * replaceNode, NodePtr targetNode);

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

      /// the master list of nodes
      NodePtrListPtr nodes_;

      /// warnings or exceptions?
      static bool strict_;
      static bool doReplaces_;
    };
  }
}

#endif

