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

#include "FWCore/ParameterSet/interface/CompositeNode.h"
#include "FWCore/ParameterSet/interface/Node.h"

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

  }
}
#endif
