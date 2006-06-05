#ifndef ParameterSet_Visitor_h
#define ParameterSet_Visitor_h

namespace edm {
  namespace pset {

    struct UsingNode;
    struct StringNode;
    struct EntryNode;
    struct VEntryNode;
    struct PSetRefNode;
    struct ContentsNode;
    class  IncludeNode;
    struct PSetNode;
    struct VPSetNode;
    struct ModuleNode;
    struct WrapperNode;
    struct OperatorNode;
    struct OperandNode;

    // ------------------------------------------------
    // simple visitor
    // only visits one level.  things that need to descend will call
    // the "acceptForChildren" method

    struct Visitor
    {
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

