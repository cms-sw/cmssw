
#include "FWCore/ParameterSet/interface/Nodes.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>


using namespace std;

namespace edm {

  namespace pset {

    Node::~Node() { }

    //--------------------------------------------------
    // UsingNode
    //--------------------------------------------------
    
    UsingNode::UsingNode(const string& name, int line) :
      Node(name, line)
    { }

    string UsingNode::type() const { return "using"; }


    void UsingNode::print(ostream& ost) const
    {
      ost << "using " << name;
    }
    
    void UsingNode::accept(Visitor& v) const
    {
      v.visitUsing(*this);
    }

    //--------------------------------------------------
    // StringNode
    //--------------------------------------------------

    StringNode::StringNode(const string& value, int line):
      Node("nameless", line),
      value_(value)      
    {  }

    string StringNode::type() const { return "string"; }

    void StringNode::print(ostream& ost) const
    {
      ost << value_;
    }

    void StringNode::accept(Visitor& v) const
    {
      v.visitString(*this);
    }


    //--------------------------------------------------
    // EntryNode
    //--------------------------------------------------

    EntryNode::EntryNode(const string& typ, const string& nam,
			 const string& val, bool track, int line):
      Node(nam, line),
      type_(typ),
      value_(val),
      tracked_(track)
    {  }

    string EntryNode::type() const { return type_; }

    void EntryNode::print(ostream& ost) const
    {
      const char* t = !tracked_? "" : "untracked ";
      ost << t << type_ << " " << name << " = " << value_;
    }

    void EntryNode::accept(Visitor& v) const
    {
      v.visitEntry(*this);
    }

    //--------------------------------------------------
    // VEntryNode
    //--------------------------------------------------

    VEntryNode::VEntryNode(const string& t, const string& n,
			   StringListPtr v,bool tr, int line):
      Node(n, line),
      type_(t),
      value_(v),
      tracked_(tr)
    { }

    string VEntryNode::type() const { return type_; }


    void VEntryNode::print(ostream& ost) const
    {
      const char* t = !tracked_ ? "" : "untracked ";
      ost << t << type_ << " " << name << " = {\n";

      if(!value_->empty())
	{
	  StringList::const_iterator ie(value_->end()),ib(value_->begin());
	  --ie;
	  copy(ib,ie,
	       ostream_iterator<string>(ost,", "));
	  ost << *ie;
	}
      ost << "\n}\n";
    }

    void VEntryNode::accept(Visitor& v) const
    {
      v.visitVEntry(*this);
    }

    //--------------------------------------------------
    // PSetRefNode
    //--------------------------------------------------

    PSetRefNode::PSetRefNode(const string& name, 
			     const string& value,
			     bool tracked,
			     int line) :
      Node(name, line),
      value_(value),
      tracked_(tracked)
    { }

    string PSetRefNode::type() const { return "PSetRef"; }


    void PSetRefNode::print(ostream& ost) const
    {
      ost << "PSet " << name << " = " << value_;
    }

    void PSetRefNode::accept(Visitor& v) const
    {
      v.visitPSetRef(*this);
    }

    //--------------------------------------------------
    // ContentsNode
    //--------------------------------------------------

    ContentsNode::ContentsNode(NodePtrListPtr value, int line):
      Node("", line),
      value_(value)
    { }

    string ContentsNode::type() const { return ""; }

    void ContentsNode::print(ostream& ost) const
    {
      ost << "{\n";
      copy(value_->begin(),value_->end(),
	   ostream_iterator<NodePtr>(ost,"\n  "));
      ost << "}\n";
    }

    void ContentsNode::accept(Visitor& v) const
    {
      v.visitContents(*this);
    }

    void ContentsNode::acceptForChildren(Visitor& v) const
    {
      NodePtrList::const_iterator i(value_->begin()),e(value_->end());
      for(;i!=e;++i)
	{
	  (*i)->accept(v);
	}
    }


    //--------------------------------------------------
    // PSetNode
    //--------------------------------------------------

    PSetNode::PSetNode(const string& t, 
		       const string& n,
		       NodePtrListPtr v,
		       bool tracked,
		       int line) :
      Node(n, line),
      type_(t),
      value_(v,line),
      tracked_(tracked)
    { }

    string PSetNode::type() const { return type_; }

    void PSetNode::print(ostream& ost) const
    {
      // if(!name.empty())
      ost << type_ << " " << name << " = ";

      value_.print(ost);
    }

    void PSetNode::accept(Visitor& v) const
    {
      v.visitPSet(*this);
      // value_.accept(v); // let the visitor for PSetNode do the children accept
    }

    void PSetNode::acceptForChildren(Visitor& v) const
    {
      //CDJ: should this be 'accept' or 'acceptForChildren' or both?
      value_.accept(v);
    }

    //--------------------------------------------------
    // VPSetNode
    //--------------------------------------------------

    VPSetNode::VPSetNode(const string& typ, 
			 const string& name,
			 NodePtrListPtr value,
			 bool tracked,
			 int line) :
      Node(name,line),
      type_(typ),
      value_(value),
      tracked_(tracked)
    { }

    string VPSetNode::type() const { return type_; }


    void VPSetNode::print(ostream& ost) const
    {
      if(value_==0) { cerr << "Badness" << endl; abort(); }

      ost << type_ << " " << name << " = {\n";
      if(!value_->empty())
	{
	  //copy(value_->begin(),value_->end(),
	  //   ostream_iterator<NodePtr>(ost,",\n  "));
	  NodePtrList::const_iterator ie(value_->end()),ib(value_->begin());
	  --ie;
	  copy(ib,ie,
	       ostream_iterator<NodePtr>(ost,", "));
	  ost << *ie;
	}  ost << "\n}\n";
  
    }

    void VPSetNode::accept(Visitor& v) const
    {
      v.visitVPSet(*this);
    }

    void VPSetNode::acceptForChildren(Visitor& v) const
    {
      NodePtrList::const_iterator i(value_->begin()),e(value_->end());
      for(;i!=e;++i)
	{
	  (*i)->accept(v);
	}
    }

    // -------------------------

    string makeOpName()
    {
      static int opcount = 0;
      ostringstream ost;
      ++opcount;
      ost << "op" << opcount;
      return ost.str();
    }

    bool operator_or_operand(NodePtr n)
    {
      Node* p = n.operator->();
      return 
	( dynamic_cast<OperatorNode*>(p) != 0 ||
	  dynamic_cast<OperandNode*>(p) != 0 );
    }

    //--------------------------------------------------
    // OperatorNode
    //--------------------------------------------------

    OperatorNode::OperatorNode(const string& type,
			       NodePtr left, 
			       NodePtr right,
			       int line):
      Node(makeOpName(), line),
      type_(type),
      left_(left),
      right_(right),
      parent_(0)
    {   
      assert( operator_or_operand(left) );
      assert( operator_or_operand(right) );
      left_->setParent(this);
      right_->setParent(this);
    }

    string OperatorNode::type() const { return type_; }


    void OperatorNode::print(ostream& ost) const
    {
      ost << "( " << left_ << " " << type_ << " " << right_ << " )";
    }


    void OperatorNode::accept(Visitor& v) const
    {
      v.visitOperator(*this);
      //throw runtime_error("OperatorNodes cannot be visited");
    }
    void OperatorNode::setParent(Node* parent){parent_=parent;}
    Node* OperatorNode::getParent(){return parent_;}

    //--------------------------------------------------
    // OperandNode
    //--------------------------------------------------

    OperandNode::OperandNode(const string& type, 
			     const string& name, 
			     int line):
      Node(name, line),
      parent_(0), 
      type_(type)
    {  }

    string OperandNode::type() const { return type_; }

    void OperandNode::print(ostream& ost) const
    {
      ost << name;
    }

    void  OperandNode::setParent(Node* parent){parent_=parent;}
    Node* OperandNode::getParent(){return parent_;}

    void OperandNode::accept(Visitor& v) const
    {
      v.visitOperand(*this);
      //throw runtime_error("OperatandNodes cannot be visited");
    }

    //--------------------------------------------------
    // WrapperNode
    //--------------------------------------------------

    WrapperNode::WrapperNode(const string& type, const string& name,
			     NodePtr w,int line):
      Node(name, line),
      type_(type),
      wrapped_(w)
    { }

    string WrapperNode::type() const { return type_; }

    void WrapperNode::print(ostream& ost) const
    {
      ost << type_ << " " << name << " = {\n"
	  << wrapped_
	  << "\n}\n";
    }

    void WrapperNode::accept(Visitor& v) const
    {
      // we do not visit lower module here, the scheduler uses those
      v.visitWrapper(*this);
    }

    //--------------------------------------------------
    // ModuleNode
    //--------------------------------------------------

    ModuleNode::ModuleNode(const string& typ, const string& instname,
			   const string& classname, NodePtrListPtr nl,
			   int line):
      Node(instname, line),
      type_(typ),
      class_(classname),
      nodes_(nl)
    { }

    string ModuleNode::type() const { return type_; }

    void ModuleNode::print(ostream& ost) const
    {
      string output_name = ( name == "nameless" ? string() : name);
      ost << type_ << " " << output_name << " = " << class_ << "\n{\n";
      copy(nodes_->begin(),nodes_->end(),
		ostream_iterator<NodePtr>(ost,"\n"));
      ost << "\n}\n";
    }

    void ModuleNode::accept(Visitor& v) const
    {
      v.visitModule(*this);
    }

    void ModuleNode::acceptForChildren(Visitor& v) const
    {
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(;i!=e;++i)
	{
	  (*i)->accept(v);
	}
    }

    // ---------------------------------------------

    Visitor::~Visitor() { }

    void Visitor::visitUsing(const UsingNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit UsingNode"); }
    void Visitor::visitString(const StringNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit StringNode"); }
    void Visitor::visitEntry(const EntryNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit EntryNode"); }
    void Visitor::visitVEntry(const VEntryNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit VEntryNode"); }
    void Visitor::visitPSetRef(const PSetRefNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit PSetRefNode"); }
    void Visitor::visitContents(const ContentsNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit ContentsNode"); }
    void Visitor::visitPSet(const PSetNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit PSetNode"); }
    void Visitor::visitVPSet(const VPSetNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit VPSetNode"); }
    void Visitor::visitModule(const ModuleNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit ModuleNode"); }
    void Visitor::visitWrapper(const WrapperNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit WrapperNode"); }
    void Visitor::visitOperator(const OperatorNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit OperatorNode"); }
    void Visitor::visitOperand(const OperandNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit OperandNode"); }

  }
}
