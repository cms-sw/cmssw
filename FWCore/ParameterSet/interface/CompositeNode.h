#ifndef ParameterSet_CompositeNode_h
#define ParameterSet_CompositeNode_h

#include "FWCore/ParameterSet/interface/Node.h"

namespace edm {
  namespace pset {

    /** CompositeNode is meant as a base class */
    class CompositeNode : public Node {
    public:
      CompositeNode(const std::string& name, NodePtrListPtr nodes, int line=-1)
      : Node(name, line), nodes_(nodes) {}

      /// deep copy
      CompositeNode(const CompositeNode & n);

      virtual void acceptForChildren(Visitor& v) const;
      virtual void print(std::ostream& ost, PrintOptions options) const;
      virtual void locate(const std::string & s, std::ostream & out) const;

      // if this is flagged as modified, all subnodes are
      virtual void setModified(bool value);
      /// if any subnodes are modified, this counts as modified
      virtual bool isModified() const;

      /// sets all children as cloned
      virtual void setCloned(bool value);

      /// also makes all subnodes register their parents
      virtual void setAsChildrensParent();

      /// finds all first-level subnodes, transparent to includes
      NodePtrListPtr children() const;

      /// finds a first-level subnode with this name
      bool findChild(const std::string & child, NodePtr & result);

      void removeChild(const std::string & child);
      void removeChild(const Node* child);

      /// returns all sub-nodes
      NodePtrListPtr nodes() const {return nodes_;}

      /// resolve any includes in sub-nodes
      virtual void resolve(std::list<std::string> & openFiles,
                           std::list<std::string> & sameLevelIncludes);

      /// if a direct descendant is a using block, inline it.
      /// otherwise, pass the call to the child nodes
      virtual void resolveUsingNodes(const NodeMap & blocks);

      /// inserts all subnodes
      virtual void insertInto(ParameterSet & pset) const;

      /// checks that no node name is duplicated
      virtual void validate() const;

    protected:
      NodePtrListPtr nodes_;
    };

  }
}

#endif

