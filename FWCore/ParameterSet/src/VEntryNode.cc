#include "FWCore/ParameterSet/interface/VEntryNode.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/parse.h"

#include <ostream>
#include <iterator>
using std::string;
using std::vector;

namespace edm {
  namespace pset {


    VEntryNode::VEntryNode(const string& t, const string& n,
                           StringListPtr v,bool tr, int line):
      Node(n, line),
      type_(t),
      value_(v),
      tracked_(tr)
    { }


     VEntryNode::VEntryNode(const VEntryNode & n)
     : Node(n),
       type_(n.type_),
       value_( new StringList(n.value_->begin() , n.value_->end()) ),
       tracked_(n.tracked_)
     {
     }

    string VEntryNode::type() const { return type_; }


    void VEntryNode::print(std::ostream& ost, Node::PrintOptions options) const
    {
      const char* t = !tracked_ ? "" : "untracked ";
      ost << t << type_ << " " << name << " = {\n  ";

      if(!value_->empty())
        {
          StringList::const_iterator ie(value_->end()),ib(value_->begin());
          --ie;
          copy(ib,ie,
               std::ostream_iterator<string>(ost,", "));
          ost << *ie;
        }
      ost << "\n  }\n";
    }

    void VEntryNode::accept(Visitor& v) const
    {
      v.visitVEntry(*this);
    }


    void VEntryNode::replaceWith(const ReplaceNode * replaceNode)
    {
      assertNotModified();
      VEntryNode * replacement = dynamic_cast<VEntryNode*>(replaceNode->value_.get());
      if(replacement == 0) {
        throw edm::Exception(errors::Configuration)
          << "Cannot replace entry vector" << name
          << " with " << replaceNode->type();
      }
      // replace the value, keep the type
      value_ = replacement->value_;
      setModified(true);
    }


    edm::Entry VEntryNode::makeEntry() const
    {
      vector<string>::const_iterator ib(value_->begin()),
        ie(value_->end()),k=ib;

     if(type()=="vstring")
       {
         vector<string> usethis;
         for(;ib!=ie;++ib) usethis.push_back(withoutQuotes(*ib));
         return Entry(usethis, !tracked_);
       }
     else if(type()=="vdouble")
       {
         vector<double> d ;
         for(ib=k;ib!=ie;++ib) d.push_back(strtod(ib->c_str(),0));
         return Entry(d, !tracked_);
       }
     else if(type()=="vint32")
       {
         vector<int> d ;
         for(ib=k;ib!=ie;++ib) d.push_back(atoi(ib->c_str()));
         return Entry(d, !tracked_);
       }
     else if(type()=="vuint32")
       {
         vector<unsigned int> d ;
         for(ib=k;ib!=ie;++ib) d.push_back(strtoul(ib->c_str(),0,10));
         return Entry(d, !tracked_);
       }
     else if(type()=="VInputTag")
       {
         vector<InputTag> d ;
         for(ib=k;ib!=ie;++ib) d.push_back( InputTag(withoutQuotes(*ib)) );
         return Entry(d, !tracked_);
       }
     else
       {
         throw edm::Exception(errors::Configuration)
           << "Bad VEntry Node type: " << type();
       }
    }

  }
}

