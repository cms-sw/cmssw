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

#include "FWCore/ParameterSet/interface/Node.h"
#include "FWCore/ParameterSet/interface/CompositeNode.h"
#include "FWCore/ParameterSet/interface/PSetNode.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/ParameterSet/interface/EntryNode.h"
#include "FWCore/ParameterSet/interface/IncludeNode.h"
#include "FWCore/ParameterSet/interface/ImplicitIncludeNode.h"
#include "FWCore/ParameterSet/interface/VEntryNode.h"
#include "FWCore/ParameterSet/interface/ModuleNode.h"
#include "FWCore/ParameterSet/interface/WrapperNode.h"
#include "FWCore/ParameterSet/interface/VPSetNode.h"


namespace edm {
  namespace pset {


    /*
      -----------------------------------------
      Usings hold: using
    */

    class UsingNode : public Node
    {
    public:
      explicit UsingNode(const std::string& name,int line=-1);
      virtual Node * clone() const { return new UsingNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost, PrintOptions options) const;
      virtual void accept(Visitor& v) const;
    };



    /*
     ------------------------------------------
     Rename: change the name to a module.  Old name no longer valid
    */

    class RenameNode : public Node
    {
    public:
      RenameNode(const std::string & type, const std::string& from,
                 const std::string & to, int line=-1)
      : Node(type, line), from_(from), to_(to) {}
      virtual Node * clone() const { return new RenameNode(*this);}
      virtual std::string type() const {return "rename";}
      virtual std::string from() const {return from_;}
      virtual std::string to() const {return to_;}
      virtual void print(std::ostream& ost, PrintOptions options) const;
      virtual void accept(Visitor& v) const;
                                                                                                          
    private:
      std::string from_;
      std::string to_;
    };

 

    /*
     ------------------------------------------
      CopyNode:  deep-copies an entire named node
    */

    class CopyNode : public Node
    {
    public:
      CopyNode(const std::string & type, const std::string& from,
                 const std::string & to, int line=-1)
      : Node(type, line), from_(from), to_(to) {}
      virtual Node * clone() const { return new CopyNode(*this);}
      virtual std::string type() const {return "copy";}
      virtual std::string from() const {return from_;}
      virtual std::string to() const {return to_;}
      virtual void print(std::ostream& ost, PrintOptions options) const;
      virtual void accept(Visitor& v) const;
                                                                                                    
    private:
      std::string from_;
      std::string to_;
    };



    /*
      -----------------------------------------
      Strings hold: a value without a name (used within VPSet)
    */

    class StringNode : public Node
    {
    public:
      explicit StringNode(const std::string& value, int line=-1);
      virtual Node * clone() const { return new StringNode(*this);}
      virtual std::string type() const;
      virtual std::string value() const {return value_;}
      virtual void print(std::ostream& ost, PrintOptions options) const;
      virtual void accept(Visitor& v) const;

    private:
      std::string value_;
    };


    /*
      -----------------------------------------
      PSetRefs hold: local name or ID of a PSet
    */

    class PSetRefNode : public Node
    {
    public:
      PSetRefNode(const std::string& name, 
		  const std::string& value,
		  bool tracked,
		  int line=-1);
      virtual Node * clone() const { return new PSetRefNode(*this);}
      virtual std::string type() const;
      std::string value() const {return value_;}
      bool isTracked() const {return tracked_;}
      virtual void print(std::ostream& ost, PrintOptions options) const;

      virtual void accept(Visitor& v) const;

    private:
      std::string value_;
      bool tracked_;
    };

    /*
      -----------------------------------------
      Contents hold: Nodes
    */

    class ContentsNode : public CompositeNode
    {
    public:
      explicit ContentsNode(NodePtrListPtr value, int line=-1);
      virtual Node * clone() const { return new ContentsNode(*this);}
      virtual std::string type() const;
      virtual void accept(Visitor& v) const;
    };

    typedef boost::shared_ptr<ContentsNode> ContentsNodePtr;


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

    class OperatorNode : public Node
    {
    public:
      OperatorNode(const std::string& t, NodePtr left, NodePtr right, int line=-1);
      /// doesn't deep-copy left & right
      virtual Node * clone() const { return new OperatorNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost, PrintOptions options) const;
      NodePtr & left() {return left_;}
      NodePtr & right() {return right_;}
      const NodePtr & left() const {return left_;}
      const NodePtr & right() const {return right_;}


      virtual void accept(Visitor& v) const;

    private:
      std::string type_;
      NodePtr left_;
      NodePtr right_;
    };

    /*
      -----------------------------------------
      Operands hold: leaf in the path expression - names of modules/sequences
    */

    class OperandNode : public Node
    {
    public:
      OperandNode(const std::string& type, const std::string& name, int line=-1);
      virtual Node * clone() const { return new OperandNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost, PrintOptions options) const;
  
      virtual void accept(Visitor& v) const;

    private:
      std::string type_;
    };


  }
}
#endif
