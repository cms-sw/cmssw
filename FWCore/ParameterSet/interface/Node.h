#ifndef ParameterSet_Node_h
#define ParameterSet_Node_h

#include <ostream>
#include <list>
#include <map>
#include <vector>
#include <string>
#include "boost/shared_ptr.hpp"

namespace edm {
  namespace pset {

    struct Visitor;
    struct ReplaceNode;
    // Base type for all nodes.  All nodes have a type associated
    // with them - this is basically the keyword that caused the
    // generation of this node.  All nodes have a name - this is the
    // name assigned to the entity

    struct Node
    {
      Node(std::string const& n, int li) : name(n), line(li), modified_(false) { }

      /// needed for deep copies
      virtual Node * clone() const = 0;

      typedef boost::shared_ptr<Node> Ptr;

      virtual std::string type() const = 0;

      virtual void  setParent(Node* /* parent */) { }
      virtual Node* getParent() { return 0; }
      virtual void print(std::ostream& ost) const = 0;
      virtual ~Node();
      virtual void accept(Visitor& v) const = 0;

      virtual void setModified(bool value) {modified_ = value;}
      /// throw an exception if this node is flagged as modified
      void assertNotModified() const;
      virtual bool isModified() const {return modified_;}
      /// throws an exception if they're not the same type
      virtual void replaceWith(const ReplaceNode * replaceNode);

      typedef std::map<std::string, Ptr> NodeMap;
      /// most subclasses won't do anything
      virtual void resolveUsingNodes(const NodeMap & blocks) {}

      std::string name;
      int         line;
      // nodes can only be modified once, so the config files can be order-independent
      bool modified_;
    };

    typedef boost::shared_ptr<Node>        NodePtr;
    typedef std::vector<std::string>       StringList;
    typedef boost::shared_ptr<StringList>  StringListPtr;
    typedef std::list<NodePtr>             NodePtrList;
    typedef boost::shared_ptr<NodePtrList> NodePtrListPtr;
    typedef NodePtrListPtr                 ParseResults;

    inline std::ostream& operator<<(std::ostream& ost, NodePtr p)
    {
      p->print(ost);
      return ost;
    }

    inline std::ostream& operator<<(std::ostream& ost, const Node& p)
    {
      p.print(ost);
      return ost;
    }

  }
}

#endif

