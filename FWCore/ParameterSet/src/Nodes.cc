
#include "FWCore/ParameterSet/interface/Nodes.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>
#include <stdexcept>

using namespace std;
namespace edm {
   namespace pset {
Node::~Node()
{
}

// ------------------

StringNode::StringNode(const string& value, int line):
  value_(value),line_(line)
{
}

std::string StringNode::type() const { return "string"; }

std::string StringNode::name() const { return "nameless"; }

void StringNode::print(std::ostream& ost) const
{
  ost << value_;
}

void StringNode::accept(Visitor& v) const
{
  v.visitString(*this);
}

// ------------------

UsingNode::UsingNode(const string& name, int line):
  name_(name), line_(line)
{
}

std::string UsingNode::type() const { return "using"; }

std::string UsingNode::name() const { return name_; }

void UsingNode::print(std::ostream& ost) const
{
  ost << "using " << name_;
}

void UsingNode::accept(Visitor& v) const
{
  v.visitUsing(*this);
}

// ------------------

EntryNode::EntryNode(const string& typ, const string& nam,
		     const string& val, bool track, int line):
  type_(typ),name_(nam),value_(val),tracked_(track),line_(line)
{
}

std::string EntryNode::type() const { return type_; }

std::string EntryNode::name() const { return name_; }

void EntryNode::print(std::ostream& ost) const
{
  const char* t = !tracked_? "" : "untracked ";
  ost << t << type_ << " " << name_ << " = " << value_;
}

void EntryNode::accept(Visitor& v) const
{
  v.visitEntry(*this);
}

// ------------------

VEntryNode::VEntryNode(const string& t, const string& n,
		       StringListPtr v,bool tr, int line):
  type_(t),name_(n),value_(v),tracked_(tr),line_(line)
{
}

std::string VEntryNode::type() const { return type_; }

std::string VEntryNode::name() const { return name_; }

void VEntryNode::print(std::ostream& ost) const
{
  const char* t = !tracked_ ? "" : "untracked ";
  ost << t << type_ << " " << name_ << " = {\n";

  if(!value_->empty())
    {
      StringList::const_iterator ie(value_->end()),ib(value_->begin());
      --ie;
      copy(ib,ie,
	   ostream_iterator<std::string>(ost,", "));
      ost << *ie;
    }
  ost << "\n}\n";
}

void VEntryNode::accept(Visitor& v) const
{
  v.visitVEntry(*this);
}

// -------------------

PSetRefNode::PSetRefNode(const string& name, const string& value,
			 int /* line */):
  name_(name), value_(value)
{
}

std::string PSetRefNode::type() const { return "PSetRef"; }

std::string PSetRefNode::name() const { return name_; }

void PSetRefNode::print(std::ostream& ost) const
{
  ost << "PSet " << name_ << " = " << value_;
}

void PSetRefNode::accept(Visitor& v) const
{
  v.visitPSetRef(*this);
}

// -------------------

ContentsNode::ContentsNode(NodePtrListPtr value, int line):
  value_(value),line_(line)
{
}

std::string ContentsNode::type() const { return ""; }
std::string ContentsNode::name() const { return ""; }

void ContentsNode::print(std::ostream& ost) const
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

// -------------------

PSetNode::PSetNode(const string& t, const string& n,
		   NodePtrListPtr v, int line):
  type_(t),name_(n),value_(v,line),line_(line)
{
}

std::string PSetNode::type() const { return type_; }

std::string PSetNode::name() const { return name_; }

void PSetNode::print(std::ostream& ost) const
{
  // if(!name_.empty())
    ost << type_ << " " << name_ << " = ";

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
// ---------------------

VPSetNode::VPSetNode(const string& typ, const string& name,
		     NodePtrListPtr value, int line):
  type_(typ),name_(name),value_(value),line_(line)
{
}

std::string VPSetNode::type() const { return type_; }

std::string VPSetNode::name() const { return name_; }

void VPSetNode::print(std::ostream& ost) const
{
  if(value_==0) { std::cerr << "Badness" << endl; abort(); }

  ost << type_ << " " << name_ << " = {\n";
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

std::string makeOpName()
{
  static int opcount = 0;
  ostringstream ost;
  ++opcount;
  ost << "op" << opcount;
  return ost.str();
}

// -------------------------

OperatorNode::OperatorNode(const string& type,NodePtr left, NodePtr right,int line):
  type_(type),name_(makeOpName()),
  left_(left),right_(right),
  parent_(0),
  line_(line)
{   
   left_->setParent(this);
   right_->setParent(this);
}

std::string OperatorNode::type() const { return type_; }

std::string OperatorNode::name() const { return name_; }

void OperatorNode::print(std::ostream& ost) const
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


// ----------------------------------

OperandNode::OperandNode(const string& type, const string& name, int line):
  parent_(0), type_(type), name_(name), line_(line)
{
  
}

std::string OperandNode::type() const { return type_; }

std::string OperandNode::name() const { return name_; }

void OperandNode::print(std::ostream& ost) const
{
  ost << name_;
}

void  OperandNode::setParent(Node* parent){parent_=parent;}
Node* OperandNode::getParent(){return parent_;}

void OperandNode::accept(Visitor& v) const
{
   v.visitOperand(*this);
   //throw runtime_error("OperatandNodes cannot be visited");
}

// -------------------------------------

WrapperNode::WrapperNode(const string& type, const string& name,
			 NodePtr w,int line):
  type_(type),name_(name),wrapped_(w),line_(line)
{
}

std::string WrapperNode::type() const { return type_; }

std::string WrapperNode::name() const { return name_; }

void WrapperNode::print(std::ostream& ost) const
{
  ost << type_ << " " << name_ << " = {\n"
      << wrapped_
      << "\n}\n";
}

void WrapperNode::accept(Visitor& v) const
{
  // we do not visit lower module here, the scheduler uses those
  v.visitWrapper(*this);
}
// -----------------------------------

ModuleNode::ModuleNode(const string& typ, const string& instname,
		       const string& classname, NodePtrListPtr nl,
		       int line):
  type_(typ),name_(instname),class_(classname),nodes_(nl),line_(line)
{
}

std::string ModuleNode::type() const { return type_; }

std::string ModuleNode::name() const { return name_; }

void ModuleNode::print(std::ostream& ost) const
{
  ost << type_ << " " << name_ << " = " << class_ << "\n{\n";
  std::copy(nodes_->begin(),nodes_->end(),
	    std::ostream_iterator<NodePtr>(ost,"\n"));
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
