
#include "FWCore/ParameterSet/interface/Nodes.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <typeinfo>
#include <iterator>

using namespace std;

namespace edm {

  namespace pset {

    //--------------------------------------------------
    // UsingNode
    //--------------------------------------------------
    
    UsingNode::UsingNode(const string& name, int line) :
      Node(name, line)
    { }

    string UsingNode::type() const { return "using"; }


    void UsingNode::print(ostream& ost, Node::PrintOptions options) const
    {
      ost << "  using " << name();
    }
    
    void UsingNode::accept(Visitor& v) const
    {
      v.visitUsing(*this);
    }


    //--------------------------------------------------
    // RenameNode
    //--------------------------------------------------
                                                                                                          
                                                                                                          
    void RenameNode::print(ostream& ost, Node::PrintOptions options) const
    {
      ost << name() << " " << from_ << " " << to_;
    }
                                                                                                          
    void RenameNode::accept(Visitor& v) const
    {
      throw edm::Exception(errors::LogicError,"Rename Nodes should always be processed by the postprocessor.  Please contact an EDM developer");
    }
                                                                                                          

    //--------------------------------------------------
    // CopyNode
    //--------------------------------------------------
                                                                                                    
                                                                                                    
    void CopyNode::print(ostream& ost, Node::PrintOptions options) const
    {
      ost << name() << " " << from_ << " " << to_;
    }
                                                                                                    
    void CopyNode::accept(Visitor& v) const
    {
      throw edm::Exception(errors::LogicError,"Rename Nodes should always be processed by the postprocessor.  Please contact an EDM developer");
    }
                                                                                                    

    //--------------------------------------------------
    // StringNode
    //--------------------------------------------------

    StringNode::StringNode(const string& value, int line):
      Node("nameless", line),
      value_(value)      
    {  }

    string StringNode::type() const { return "string"; }

    void StringNode::print(ostream& ost, Node::PrintOptions options) const
    {
      ost <<  value_;
    }

    void StringNode::accept(Visitor& v) const
    {
      v.visitString(*this);
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


    void PSetRefNode::print(ostream& ost, Node::PrintOptions options) const
    {
      ost << "PSet " << name() << " = " << value_;
    }

    void PSetRefNode::accept(Visitor& v) const
    {
      v.visitPSetRef(*this);
    }

    //--------------------------------------------------
    // ContentsNode
    //--------------------------------------------------

    ContentsNode::ContentsNode(NodePtrListPtr value, int line):
      CompositeNode("", value, line)
    { }

    string ContentsNode::type() const { return ""; }

    void ContentsNode::accept(Visitor& v) const
    {
      v.visitContents(*this);
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
      right_(right)
    {   
      assert( operator_or_operand(left) );
      assert( operator_or_operand(right) );
      left_->setParent(this);
      right_->setParent(this);
    }

    string OperatorNode::type() const { return type_; }


    void OperatorNode::print(ostream& ost, Node::PrintOptions options) const
    {
      ost << "( " << left_ << " " << type_ << " " << right_ << " )";
    }


    void OperatorNode::accept(Visitor& v) const
    {
      v.visitOperator(*this);
      //throw runtime_error("OperatorNodes cannot be visited");
    }
    //--------------------------------------------------
    // OperandNode
    //--------------------------------------------------

    OperandNode::OperandNode(const string& type, 
			     const string& name, 
			     int line):
      Node(name, line),
      type_(type)
    {  }

    string OperandNode::type() const { return type_; }

    void OperandNode::print(ostream& ost, Node::PrintOptions options) const
    {
      ost << name();
    }

    void OperandNode::accept(Visitor& v) const
    {
      v.visitOperand(*this);
      //throw runtime_error("OperatandNodes cannot be visited");
    }

  }
}
