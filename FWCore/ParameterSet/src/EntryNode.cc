#include "FWCore/ParameterSet/interface/EntryNode.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/Utilities/interface/EDMException.h"

using std::string;

namespace edm {
  namespace pset {


    EntryNode::EntryNode(const string& typ, const string& nam,
                         const string& val, bool track, int line):
      Node(nam, line),
      type_(typ),
      value_(val),
      tracked_(track)
    {  }


    string EntryNode::type() const { return type_; }


    void EntryNode::print(std::ostream& ost, Node::PrintOptions options) const
    {
      const char* t = !tracked_? "" : "untracked ";
      ost << t << type_ << " " << name << " = " << value_;
    }


    void EntryNode::accept(Visitor& v) const
    {
      v.visitEntry(*this);
    }


    void EntryNode::replaceWith(const ReplaceNode * replaceNode) {
      assertNotModified();
      EntryNode * replacement = dynamic_cast<EntryNode*>(replaceNode->value_.get());
      if(replacement == 0) {
        throw edm::Exception(errors::Configuration)
          << "Cannot replace entry " << name
          << " with " << replaceNode->type();
      }
      // replace the value, keep the type
      value_ = replacement->value_;
      setModified(true);
    }


    Entry EntryNode::makeEntry() const
    {
      if(type()=="string")
       {
         string usethis(withoutQuotes(value_));
         return Entry(usethis, !tracked_);
       }
     else if (type()=="FileInPath")
       {
         edm::FileInPath fip(withoutQuotes(value_));
         return Entry(fip, !tracked_);
       }
     else if (type()=="InputTag")
       {
         edm::InputTag tag(value_);
         return Entry(tag, !tracked_);
       }
     else if(type()=="double")
       {
         double d = strtod(value_.c_str(),0);
         return Entry(d, !tracked_);
       }
     else if(type()=="int32")
       {
         int d = strtol(value_.c_str(),0,0);
         return Entry(d, !tracked_);
       }
     else if(type()=="uint32")
       {
         unsigned int d = strtoul(value_.c_str(),0,0);
         return Entry(d, !tracked_);
       }
     else if(type()=="bool")
       {
         bool d(false);
         if(value_=="true" || value_=="T" || value_=="True" ||
            value_=="1" || value_=="on" || value_=="On")
           d = true;

         return Entry(d, !tracked_);
       }
     else
       {
         throw edm::Exception(errors::Configuration)
           << "Bad Entry Node type: " << type();
       }

     }

  }
}

