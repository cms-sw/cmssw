#ifndef ParameterSet_OperatorNode_h
#define ParameterSet_OperatorNode_h


#include "FWCore/ParameterSet/interface/Node.h"


namespace edm {
  namespace pset {

    /*
      -----------------------------------------
      Operators hold: and/comma type, left and right operands, which
      are modules/sequences or more operators
    */

    class OperatorNode : public Node
    {
    public:
      OperatorNode(const std::string& t, NodePtr left, NodePtr right, int line=-1);
      /// deep-copy left & right
      virtual Node * clone() const;
      virtual std::string type() const;
      virtual void print(std::ostream& ost, PrintOptions options) const;
      virtual bool findChild(const std::string & childName, NodePtr & result);
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


  }
}
#endif
