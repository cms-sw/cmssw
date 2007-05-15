#ifndef ParameterSet_OperandNode_h
#define ParameterSet_OperandNode_h


#include "FWCore/ParameterSet/interface/Node.h"


namespace edm {
  namespace pset {

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
