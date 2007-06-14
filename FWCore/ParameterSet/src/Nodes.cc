
#include "FWCore/ParameterSet/interface/Nodes.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iosfwd>

namespace edm {

  namespace pset {

    //--------------------------------------------------
    // UsingNode
    //--------------------------------------------------
    
    UsingNode::UsingNode(const std::string& name, int line) :
      Node(name, line)
    { }

    std::string UsingNode::type() const { return "using"; }


    void UsingNode::print(std::ostream& ost, Node::PrintOptions options) const
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
                                                                                                          
                                                                                                          
    void RenameNode::print(std::ostream& ost, Node::PrintOptions options) const
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
                                                                                                    
                                                                                                    
    void CopyNode::print(std::ostream& ost, Node::PrintOptions options) const
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

    StringNode::StringNode(const std::string& value, int line):
      Node("nameless", line),
      value_(value)      
    {  }

    std::string StringNode::type() const { return "string"; }

    void StringNode::print(std::ostream& ost, Node::PrintOptions options) const
    {
      ost <<  value_;
    }

    void StringNode::accept(Visitor& v) const
    {
      v.visitString(*this);
    }


    //--------------------------------------------------
    // ContentsNode
    //--------------------------------------------------

    ContentsNode::ContentsNode(NodePtrListPtr value, int line):
      CompositeNode("", value, line)
    { }

    std::string ContentsNode::type() const { return ""; }

    void ContentsNode::accept(Visitor& v) const
    {
      v.visitContents(*this);
    }


  }
}
