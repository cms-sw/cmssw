
#include "FWCore/ParameterSet/interface/OperatorNode.h"
#include "FWCore/ParameterSet/interface/OperandNode.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iosfwd>

namespace edm {

  namespace pset {


    // -------------------------

    std::string makeOpName()
    {
      static int opcount = 0;
      std::ostringstream ost;
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

    OperatorNode::OperatorNode(const std::string& type,
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

    std::string OperatorNode::type() const { return type_; }


    Node * OperatorNode::clone() const 
    { 
      OperatorNode * newNode =  new OperatorNode(*this);
      newNode->left_ = NodePtr( left_->clone() );
      newNode->right_ = NodePtr( right_->clone() );
      newNode->left_->setParent(newNode);
      newNode->right_->setParent(newNode);
      return newNode;
    }


    void OperatorNode::print(std::ostream& ost, Node::PrintOptions options) const
    {
      ost << "( " << left_ << " " << type_ << " " << right_ << " )";
    }


    bool OperatorNode::findChild(const std::string & childName, NodePtr & result)
    {
      bool foundLeft  = left()->findChild(childName, result);
      bool foundRight = right()->findChild(childName, result);

      if(foundLeft && foundRight)
      {
        throw edm::Exception(errors::Configuration)
         << "A child named " << childName 
         << " was found on both sides of an operator"
         << "\nfrom " << traceback();
      }
      return (foundLeft || foundRight);
    }
        

    void OperatorNode::accept(Visitor& v) const
    {
      v.visitOperator(*this);
      //throw runtime_error("OperatorNodes cannot be visited");
    }

  }
}
