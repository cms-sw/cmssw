#ifndef FWCore_ParameterSet_Node_h
#define FWCore_ParameterSet_Node_h

#include <iosfwd>
#include <list>
#include <map>
#include <vector>
#include <string>
#include "boost/shared_ptr.hpp"

namespace edm {

  class Entry;
  class ParameterSet;
  class ProcessDesc;

  namespace pset {

    class Visitor;
    class ReplaceNode;

    // Base type for all nodes.  All nodes have a type associated
    // with them - this is basically the keyword that caused the
    // generation of this node.  All nodes have a name - this is the
    // name assigned to the entity

    class Node
    {
    public:
      Node(std::string const& n, int li);

      std::string name() const {return name_;}
      void setName(const std::string & n) {name_ = n;}
      int line() const {return line_;}

      /// needed for deep copies
      virtual Node * clone() const = 0;

      typedef boost::shared_ptr<Node> Ptr;

      virtual std::string type() const = 0;

      virtual void  setParent(Node*  parent) { parent_ = parent;}
      virtual Node* getParent() const { return parent_; }
      bool isInclude() const;
      /// the IncludeNode parent
      virtual const Node * getIncludeParent() const;
      /// some people care whether the parent was cfi, cff, or cfg
      std::string includeParentSuffix() const;
           

      /// leaf nodes won't do anything
      virtual void setAsChildrensParent() {}
      /// finds a first-level subnode with this name
      /// leaf nodes will return false
      virtual bool findChild(const std::string & child, boost::shared_ptr<Node> & result) {return false;}
      /// throw an exception if your contents aren't valid
      virtual void validate() const {}


      /// shows a dot-delimited path of all modules & PSets including this Node
      virtual void dotDelimitedPath(std::string & path) const;
      /// Shows where this was included from
      /// default passes up to parent
      virtual void printTrace(std::ostream& ost) const;
      /// returns a trace of all the nodes above this one
      std::string traceback() const;

      /// will print a trace if the node matches.
      /// good for searching through deep include hierarchies
      virtual void locate(const std::string & s, std::ostream& ost) const;

      /// whether or not to expand the contents of included files
      enum PrintOptions { COMPRESSED, EXPANDED };
      virtual void print(std::ostream& ost, PrintOptions options=COMPRESSED) const = 0;
      virtual ~Node();
      virtual void accept(Visitor& v) const = 0;

      virtual void setModified(bool value);
      virtual bool isModified() const {return modified_;}

      /// sometimes it matters whether the node was introduced via cloning
      virtual void setCloned(bool value) {cloned_ = value;}
      virtual bool isCloned() const {return cloned_;}

      /// throws by default
      virtual bool isTracked() const {return false;}
      virtual bool isReplaceable() const {return false;}
      virtual void replaceWith(const ReplaceNode * replaceNode);

      typedef std::map<std::string, Ptr> NodeMap;
      /// most subclasses won't do anything
      virtual void resolve(std::list<std::string> & openFiles, 
                           std::list<std::string> & sameLevelIncludes,
                           bool strict) {}
      virtual void resolveUsingNodes(const NodeMap & blocks, bool strict) {}

      /// Nodes which can exist on the top level of the
      /// parse tree should implement this
      virtual void insertInto(ProcessDesc & procDesc) const;
      /// Nodes which can be inserted into ParameterSets
      /// which aren't top-level processes should overload this.
      virtual void insertInto(edm::ParameterSet & pset) const;
      /// makes a ParameterSet Entry for this Node
      virtual edm::Entry makeEntry() const;


    private:
      std::string name_;
      int         line_;
      // nodes can only be modified once, so the config files can be order-independent
      bool modified_;
      bool cloned_;
      Node * parent_;
    };

    typedef boost::shared_ptr<Node>        NodePtr;
    typedef std::vector<std::string>       StringList;
    typedef boost::shared_ptr<StringList>  StringListPtr;
    typedef std::list<NodePtr>             NodePtrList;
    typedef boost::shared_ptr<NodePtrList> NodePtrListPtr;
    typedef NodePtrListPtr                 ParseResults;

    inline std::ostream& operator<<(std::ostream& ost, NodePtr p)
    {
      p->print(ost, Node::COMPRESSED);
      return ost;
    }

    inline std::ostream& operator<<(std::ostream& ost, const Node& p)
    {
      p.print(ost, Node::COMPRESSED);
      return ost;
    }

  }
}

#endif

