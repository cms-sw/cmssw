#ifndef ParameterSet_EntryNode_h
#define ParameterSet_EntryNode_h

#include "FWCore/ParameterSet/interface/Node.h"

namespace edm {
  namespace pset {

    /*
      -----------------------------------------
      Entries hold: bool, int32, uint32, double, string, and using
      This comment may be wrong. 'using' statements create UsingNodes,
      according to the rules in pset_parse.y
    */

    struct EntryNode : public Node
    {
      EntryNode(const std::string& type, const std::string& name,
                const std::string& values, bool tracked, int line=-1);
      virtual Node * clone() const { return new EntryNode(*this);}
      virtual std::string type() const;
      virtual void print(std::ostream& ost) const;

      virtual void accept(Visitor& v) const;
      // keeps the orignal type and tracked-ness
      virtual void replaceWith(const ReplaceNode *);
      /// the components of ParameterSets
      virtual edm::Entry makeEntry() const;

      std::string type_;
      std::string value_;
      bool tracked_;
    };
  }
}

#endif

