#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  namespace pset {

    Visitor::~Visitor() { }

    void Visitor::visitUsing(const UsingNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit UsingNode"); }
    void Visitor::visitString(const StringNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit StringNode"); }
    void Visitor::visitEntry(const EntryNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit EntryNode"); }
    void Visitor::visitVEntry(const VEntryNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit VEntryNode"); }
    void Visitor::visitContents(const ContentsNode&)
    { throw edm::Exception(errors::LogicError,"attempt to visit ContentsNode"); }
    void Visitor::visitInclude(const IncludeNode &)
    { throw edm::Exception(errors::LogicError,"attempt to visit IncludeNode"); }
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

