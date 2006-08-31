#ifndef ParameterSet_Visitor_h
#define ParameterSet_Visitor_h

namespace edm {
  namespace pset {

    class UsingNode;
    class StringNode;
    class EntryNode;
    class VEntryNode;
    class PSetRefNode;
    class ContentsNode;
    class  IncludeNode;
    class PSetNode;
    class VPSetNode;
    class ModuleNode;
    class WrapperNode;
    class OperatorNode;
    class OperandNode;

    // ------------------------------------------------
    // simple visitor
    // only visits one level.  things that need to descend will call
    // the "acceptForChildren" method

    class Visitor
    {
    public:
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
      virtual void visitInclude(const IncludeNode &);
      virtual void visitWrapper(const WrapperNode&);
      virtual void visitOperator(const OperatorNode&); //may not be needed
      virtual void visitOperand(const OperandNode&); //may not be needed
    };
  }
}

#endif

