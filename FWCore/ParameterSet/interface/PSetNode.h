#ifndef ParameterSet_PSetNode_h
#define ParameterSet_PSetNode_h

#include "FWCore/ParameterSet/interface/CompositeNode.h"

namespace edm {
  namespace pset {

    class PSetNode : public CompositeNode
    {
    public:
      PSetNode(const std::string& typ,
               const std::string& name,
               NodePtrListPtr value,
               bool tracked,
               int line=-1);
      virtual Node * clone() const { return new PSetNode(*this);}
      virtual std::string type() const;
      bool isTracked() const {return tracked_;}
      virtual void print(std::ostream& ost, PrintOptions options) const;
      virtual bool isModified() const;

      virtual void accept(Visitor& v) const;
      virtual void replaceWith(const ReplaceNode * replaceNode);

      /// makes an entry for a ParametersSet object
      virtual edm::Entry makeEntry() const;
      /// insert into a higher parameterset
      virtual void insertInto(edm::ParameterSet & pset) const;
      /// Insert into the top level of the tree
      virtual void insertInto(edm::ProcessDesc & procDesc) const;
      /// creates a ProcessDesc and fills it.  Assumes this is
      /// a node of type "process"
      void fillProcess(edm::ProcessDesc & procDesc) const;

    private:
      std::string type_;
      bool tracked_;
    };

  }
}

#endif

