#ifndef PARAMETERSET_NODES_H
#define PARAMETERSET_NODES_H

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

#include "boost/shared_ptr.hpp"
namespace edm {
   namespace pset {
/*
  ----------------------------------------
  base class for all nodes.
  All nodes have a type associated with them - this is basically
  the keyword that caused the generation of this node.
  All nodes have a name - this is the name assigned to the entity
 */

struct Visitor;

struct Node
{
  virtual std::string type() const = 0;
  virtual std::string name() const = 0;
  virtual void print(std::ostream& ost) const = 0;
  virtual ~Node();
  virtual void accept(Visitor& v) const = 0;
};


/*
  -----------------------------------------
  standard ways that nodes are used and other things
*/

typedef std::vector<std::string> StringList;
typedef boost::shared_ptr<StringList> StringListPtr;
typedef boost::shared_ptr<Node> NodePtr;
typedef std::list<NodePtr> NodePtrList;
typedef boost::shared_ptr<NodePtrList> NodePtrListPtr;

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

/*
  -----------------------------------------
  Usings hold: using
*/

struct UsingNode : public Node
{
  explicit UsingNode(const std::string& name,int line=-1);
  virtual std::string type() const;
  virtual std::string name() const;
  virtual void print(std::ostream& ost) const;

  virtual void accept(Visitor& v) const;

  std::string name_;
  int line_;
};

/*
  -----------------------------------------
  Strings hold: a value without a name (used within VPSet)
*/

struct StringNode : public Node
{
  explicit StringNode(const std::string& value, int line=-1);
  virtual std::string type() const;
  virtual std::string name() const;
  virtual void print(std::ostream& ost) const;

  virtual void accept(Visitor& v) const;

  std::string value_;
  int line_;
};

/*
  -----------------------------------------
  Entries hold: bool, int32, uint32, double, string, and using
*/

struct EntryNode : public Node
{
  EntryNode(const std::string& type, const std::string& name,
	    const std::string& values, bool tracked, int line=-1);
  virtual std::string type() const;
  virtual std::string name() const;
  virtual void print(std::ostream& ost) const;

  virtual void accept(Visitor& v) const;

  std::string type_;
  std::string name_;
  std::string value_;
  bool tracked_;
  int line_;
};

/*
  -----------------------------------------
  VEntries hold: vint32, vuint32, vdouble, vstring
 */

struct VEntryNode : public Node
{
  VEntryNode(const std::string& typ, const std::string& name,
	     StringListPtr values,bool tracked, int line=-1);
  virtual std::string type() const;
  virtual std::string name() const;
  virtual void print(std::ostream& ost) const;

  virtual void accept(Visitor& v) const;

  std::string type_;
  std::string name_;
  StringListPtr value_;
  bool tracked_;
  int line_;
};

/*
  -----------------------------------------
  PSetRefs hold: local name or ID of a PSet
*/

struct PSetRefNode : public Node
{
  PSetRefNode(const std::string& name, const std::string& value,
	      int line=-1);
  virtual std::string type() const;
  virtual std::string name() const;
  virtual void print(std::ostream& ost) const;

  virtual void accept(Visitor& v) const;

  std::string name_;
  std::string value_;
  int line_;
};

/*
  -----------------------------------------
  Contents hold: Nodes
 */

struct ContentsNode : public Node
{
  explicit ContentsNode(NodePtrListPtr value, int line=-1);
  virtual std::string type() const;
  virtual std::string name() const;
  virtual void print(std::ostream& ost) const;

  virtual void accept(Visitor& v) const;
  void acceptForChildren(Visitor& v) const;

  NodePtrListPtr value_;
  int line_;
};

typedef boost::shared_ptr<ContentsNode> ContentsNodePtr;

/*
  -----------------------------------------
  PSets hold: Contents
 */

struct PSetNode : public Node
{
  PSetNode(const std::string& typ, const std::string& name,
	   NodePtrListPtr value, int line=-1);
  virtual std::string type() const;
  virtual std::string name() const;
  virtual void print(std::ostream& ost) const;

  virtual void accept(Visitor& v) const;
  void acceptForChildren(Visitor& v) const;

  std::string type_;
  std::string name_;
  ContentsNode value_;
  int line_;
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

struct VPSetNode : public Node
{
  VPSetNode(const std::string& typ, const std::string& name,
	    NodePtrListPtr value, int line=-1);
  virtual std::string type() const;
  virtual std::string name() const;
  virtual void print(std::ostream& ost) const;

  virtual void accept(Visitor& v) const;
  void acceptForChildren(Visitor& v) const;

  std::string type_;
  std::string name_;
  NodePtrListPtr value_;
  int line_;
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
  OperatorNode(const std::string& t, NodePtr left, NodePtr right,
	       int line=-1);

  virtual std::string type() const;
  virtual std::string name() const;
  virtual void print(std::ostream& ost) const;

  virtual void accept(Visitor& v) const;

  std::string type_;
  std::string name_;
  NodePtr left_;
  NodePtr right_;
  int line_;
};

/*
  -----------------------------------------
  Operands hold: leaf in the path expression - names of modules/sequences
 */

struct OperandNode : public Node
{
  OperandNode(const std::string& type, const std::string& name, int line=-1);
  virtual std::string type() const;
  virtual std::string name() const;
  virtual void print(std::ostream& ost) const;

  virtual void accept(Visitor& v) const;

  std::string type_;
  std::string name_;
  int line_;
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

  virtual std::string type() const;
  virtual std::string name() const;
  virtual void print(std::ostream& ost) const;

  virtual void accept(Visitor& v) const;

  std::string type_;
  std::string name_;
  NodePtr wrapped_;
  int line_;
};

/*
  -----------------------------------------
  Modules hold: source (named/unnamed), modules
 */

struct ModuleNode : public Node
{
  ModuleNode(const std::string& type, const std::string& instname,
	     const std::string& classname,
	     NodePtrListPtr nl,int line=-1);

  virtual std::string type() const;
  virtual std::string name() const;
  virtual void print(std::ostream& ost) const;

  virtual void accept(Visitor& v) const;
  void acceptForChildren(Visitor& v) const;

  std::string type_;
  std::string name_;
  std::string class_;
  NodePtrListPtr nodes_;
  int line_;
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
