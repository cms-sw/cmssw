#ifndef ParameterSet_Nodes_h
#define ParameterSet_Nodes_h

/*
  The parser output is a tree of Nodes containing unprocessed strings.

  Ownership rule: Objects passed by pointer into constructors are owned
  by the newly created object (defined by the classes below).  In other
  words ownership is transferred to these objects.  Furthermore, all
  object lifetimes are managed using boost share_ptr.

  The name method needs to be stripped out.  They are not useful
*/

#include <algorithm>
#include <iterator>
#include <ostream>
#include <sstream>
#include <list>
#include <vector>
#include <string>

//RICK TEMP
#include <iostream>

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


    /** CompositeNode is meant as a base class */
    struct CompositeNode : public Node {
      CompositeNode(const std::string& name, NodePtrListPtr nodes, int line=-1)
      : Node(name, line), nodes_(nodes) {}

      /// deep copy
      CompositeNode(const CompositeNode & n);
     
      virtual void acceptForChildren(Visitor& v) const;
      virtual void print(std::ostream& ost) const;
      // if this is flagged as modified, all subnodes are
      virtual void setModified(bool value);
      /// if any subnodes are modified, this counts as modified
      virtual bool isModified() const;
      /// finds a first-level subnode with this name
      NodePtr findChild(const std::string & child);

      /// returns all sub-nodes
      NodePtrListPtr nodes() const {return nodes_;}
      

      NodePtrListPtr nodes_;
    };



    /*
      -----------------------------------------
      Usings hold: using
    */

    struct UsingNode : public Node
    {
      explicit UsingNode(const std::string& name,int line=-1);
      virtual Node * clone() const { return new UsingNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;
      virtual void accept(Visitor& v) const;
    };


   /*
      -----------------------------------------
      Replace : instructions for replacing the value of this node 
    */

    struct ReplaceNode : public Node
    {
      ReplaceNode(const std::string & type, const std::string& name, 
                  NodePtr value, int line=-1)
      : Node(name, line), type_(type), value_(value) {}
      /// deep copy
      ReplaceNode(const ReplaceNode & n);
      virtual Node * clone() const { return new ReplaceNode(*this);}
      virtual std::string type() const {return type_;}
      virtual void print(std::ostream& ost) const;
      virtual void accept(Visitor& v) const;

      std::string type_;
      NodePtr value_;
    };


    /*
     ------------------------------------------
     Rename: change the name to a module.  Old name no longer valid
    */

    struct RenameNode : public Node
    {
      RenameNode(const std::string & type, const std::string& from,
                 const std::string & to, int line=-1)
      : Node(type, line), from_(from), to_(to) {}
      virtual Node * clone() const { return new RenameNode(*this);}
      virtual std::string type() const {return "rename";}
      virtual std::string from() const {return from_;}
      virtual std::string to() const {return to_;}
      virtual void print(std::ostream& ost) const;
      virtual void accept(Visitor& v) const;
                                                                                                          
      std::string from_;
      std::string to_;
    };

 

    /*
     ------------------------------------------
      CopyNode:  deep-copies an entire named node
    */

    struct CopyNode : public Node
    {
      CopyNode(const std::string & type, const std::string& from,
                 const std::string & to, int line=-1)
      : Node(type, line), from_(from), to_(to) {}
      virtual Node * clone() const { return new CopyNode(*this);}
      virtual std::string type() const {return "copy";}
      virtual std::string from() const {return from_;}
      virtual std::string to() const {return to_;}
      virtual void print(std::ostream& ost) const;
      virtual void accept(Visitor& v) const;
                                                                                                    
      std::string from_;
      std::string to_;
    };



    /*
      -----------------------------------------
      Strings hold: a value without a name (used within VPSet)
    */

    struct StringNode : public Node
    {
      explicit StringNode(const std::string& value, int line=-1);
      virtual Node * clone() const { return new StringNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;
      virtual void accept(Visitor& v) const;

      std::string value_;
    };

    /*
      -----------------------------------------
      Entries hold: bool, int32, uint32, double, string, and using
      This comment may be wrong. 'using' statements create UsingNodes,
      according to the rules in pset_parse.y
    */

    struct EntryNode : public Node
    {
      EntryNode(const std::string& type, const std::string& name,
		const std::string& values, bool tracked, int line=-1);
      virtual Node * clone() const { return new EntryNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;

      virtual void accept(Visitor& v) const;
      // keeps the orignal type and tracked-ness
      virtual void replaceWith(const ReplaceNode *);

      std::string type_;
      std::string value_;
      bool tracked_;
    };

    /*
      -----------------------------------------
      VEntries hold: vint32, vuint32, vdouble, vstring
    */

    struct VEntryNode : public Node
    {
      VEntryNode(const std::string& typ, const std::string& name,
		 StringListPtr values,bool tracked, int line=-1);
      /// deep copy
      VEntryNode(const VEntryNode & n);
      virtual Node * clone() const { return new VEntryNode(*this);}

      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;

      virtual void accept(Visitor& v) const;
      // keeps the orignal type and tracked-ness
      virtual void replaceWith(const ReplaceNode *);


      std::string type_;
      StringListPtr value_;
      bool tracked_;
    };

    /*
      -----------------------------------------
      PSetRefs hold: local name or ID of a PSet
    */

    struct PSetRefNode : public Node
    {
      PSetRefNode(const std::string& name, 
		  const std::string& value,
		  bool tracked,
		  int line=-1);
      virtual Node * clone() const { return new PSetRefNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;

      virtual void accept(Visitor& v) const;

      std::string value_;
      bool tracked_;
    };

    /*
      -----------------------------------------
      Contents hold: Nodes
    */

    struct ContentsNode : public CompositeNode
    {
      explicit ContentsNode(NodePtrListPtr value, int line=-1);
      virtual Node * clone() const { return new ContentsNode(*this);}
      virtual std::string type() const;
      virtual void accept(Visitor& v) const;
    };

    typedef boost::shared_ptr<ContentsNode> ContentsNodePtr;

    /*
      -----------------------------------------
      PSets hold: Contents
    */

    struct PSetNode : public Node
    {
      PSetNode(const std::string& typ, 
	       const std::string& name,
	       NodePtrListPtr value,
	       bool tracked,
	       int line=-1);
      virtual Node * clone() const { return new PSetNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;
      virtual bool isModified() const;

      virtual void accept(Visitor& v) const;
      void acceptForChildren(Visitor& v) const;
      virtual void replaceWith(const ReplaceNode * replaceNode);

      std::string type_;
      ContentsNode value_;
      bool tracked_;
    };

    /*
      -----------------------------------------
      Extra things we need
    */

    typedef boost::shared_ptr<PSetNode> PSetNodePtr;
    typedef std::list<PSetNodePtr> PSetNodePtrList;
    typedef boost::shared_ptr<PSetNodePtrList> PSetNodePtrListPtr;

    /*
      -----------------------------------------
      VPSets hold: ParameterSet nodes or ParameterSet names/IDs stored in Entries
    */

    struct VPSetNode : public CompositeNode
    {
      VPSetNode(const std::string& typ, 
		const std::string& name,
		NodePtrListPtr value,
		bool tracked,
		int line=-1);
      virtual Node * clone() const { return new VPSetNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;

      virtual void accept(Visitor& v) const;

      std::string type_;
      bool tracked_;
    };

    /*
      -----------------------------------------
      utility to create a unique name for the operator nodes
    */

    std::string makeOpName();

    /*
      -----------------------------------------
      Operators hold: and/comma type, left and right operands, which
      are modules/sequences or more operators
    */

    struct OperatorNode : public Node
    {
      OperatorNode(const std::string& t, NodePtr left, NodePtr right, int line=-1);
      /// doesn't deep-copy left & right
      virtual Node * clone() const { return new OperatorNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;

      virtual void accept(Visitor& v) const;

      virtual void  setParent(Node* parent);
      virtual Node* getParent(); 

      std::string type_;
      NodePtr left_;
      NodePtr right_;
      Node*   parent_;
    };

    /*
      -----------------------------------------
      Operands hold: leaf in the path expression - names of modules/sequences
    */

    struct OperandNode : public Node
    {
      OperandNode(const std::string& type, const std::string& name, int line=-1);
      virtual Node * clone() const { return new OperandNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;
  
      virtual void accept(Visitor& v) const;

      virtual void    setParent(Node* parent); 
      virtual Node*   getParent(); 

      Node* parent_;
      std::string type_;
    };

    /*
      -----------------------------------------
      Wrappers hold: sequences, paths, endpaths
      They hold another Node that actually contains the information.
    */

    struct WrapperNode : public Node
    {
      WrapperNode(const std::string& type, const std::string& name,
		  NodePtr w, int line=-1);
      virtual Node * clone() const { return new WrapperNode(*this);}

      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;

      virtual void accept(Visitor& v) const;

      std::string type_;
      NodePtr wrapped_;
    };

    /*
      -----------------------------------------
      Modules hold: source (named/unnamed), modules
    */

    struct ModuleNode : public CompositeNode
    {
      ModuleNode(const std::string& type, const std::string& instname,
		 const std::string& classname,
		 NodePtrListPtr nl,int line=-1);
      virtual Node * clone() const { return new ModuleNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;

      virtual void accept(Visitor& v) const;
      virtual void replaceWith(const ReplaceNode * replaceNode);

      std::string type_;
      std::string class_;
    };

    // ------------------------------------------------
    // simple visitor
    // only visits one level.  things that need to descend will call
    // the "acceptForChildren" method

    struct Visitor
    {
      virtual ~Visitor();

      virtual void visitUsing(const UsingNode&);
      virtual void visitString(const StringNode&);
      virtual void visitEntry(const EntryNode&);
      virtual void visitVEntry(const VEntryNode&);
      virtual void visitPSetRef(const PSetRefNode&);
      virtual void visitContents(const ContentsNode&);
      virtual void visitPSet(const PSetNode&);
      virtual void visitVPSet(const VPSetNode&);
      virtual void visitModule(const ModuleNode&);
      virtual void visitWrapper(const WrapperNode&);
      virtual void visitOperator(const OperatorNode&); //may not be needed
      virtual void visitOperand(const OperandNode&); //may not be needed
    };
  }
}
#endif
