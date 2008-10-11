#include "FWCore/ParameterSet/interface/EntryNode.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <boost/cstdint.hpp>

#include <iosfwd>
#include <iostream>
namespace edm {
  namespace pset {


    EntryNode::EntryNode(const std::string& typ, const std::string& nam,
                         const std::string& val, bool untracked, int line):
      Node(nam, line),
      type_(typ),
      value_(val),
      tracked_(!untracked)
    {  }


    std::string EntryNode::type() const { return type_; }


    void EntryNode::print(std::ostream& ost, Node::PrintOptions options) const
    {
      const char* t = tracked_? "" : "untracked ";
      ost << t << type() << " " << name() << " = " << value();
    }


    void EntryNode::locate(const std::string & s, std::ostream & out) const
    {
      std::string match = "";
      if( value().find(s,0) != std::string::npos)
      {
        match = value();
      }
      if( name().find(s,0) != std::string::npos)
      {
        match = name();
      }

      if( match != "" )
      {
        print(out, COMPRESSED);
        out << std::endl;
        printTrace(out);
        out << std::endl;
      }
    }

    void EntryNode::accept(Visitor& v) const
    {
      v.visitEntry(*this);
    }


    void EntryNode::replaceWith(const ReplaceNode * replaceNode) {
      EntryNode * replacement = replaceNode->value<EntryNode>();
      if(replacement == 0) {
        throw edm::Exception(errors::Configuration)
          << "Cannot replace entry " << name()
          << " with " << replaceNode->type();
      }
      // replace the value, keep the type
      value_ = replacement->value_;
      setModified(true);
    }


    Entry EntryNode::makeEntry() const
    {
      // for checks of strtowhatever
      char * end;
      if(type()=="string") {
         std::string usethis(withoutQuotes(value_));
         return Entry(name(), usethis, tracked_);
     }
     else if (type()=="FileInPath") {
         edm::FileInPath fip(withoutQuotes(value_));
         return Entry(name(), fip, tracked_);
     }
     else if (type()=="InputTag") {
         edm::InputTag tag(withoutQuotes(value_));
         return Entry(name(), tag, tracked_);
     }
     else if (type()=="EventID") {
         // decodes, then encodes again
         edm::EventID eventID;
         edm::decode(eventID, value_);
         return Entry(name(), eventID, tracked_);
     }
     else if (type()=="LuminosityBlockID") {
         // decodes, then encodes again
         edm::LuminosityBlockID lumiID;
         edm::decode(lumiID, value_);
         return Entry(name(), lumiID, tracked_);
     }
     else if(type()=="double") {
         double d = strtod(value_.c_str(),&end);
         checkParse(value_, end);
         return Entry(name(), d, tracked_);
     }
     else if(type()=="int32") {
         int d = strtol(value_.c_str(),&end,0);
         checkParse(value_, end);
         return Entry(name(), d, tracked_);
     }
     else if(type()=="uint32") {
         unsigned int d = strtoul(value_.c_str(),&end,0);
         checkParse(value_, end);
         return Entry(name(), d, tracked_);
     }
     else if(type()=="int64") {
         boost::int64_t d = strtol(value_.c_str(),&end,0);
         checkParse(value_, end);
         return Entry(name(), d, tracked_);
     }
     else if(type()=="uint64") {
         boost::uint64_t d = strtoul(value_.c_str(),&end,0);
         checkParse(value_, end);
         return Entry(name(), d, tracked_);
     }
     else if(type()=="bool") {
         bool d(false);
         if(value_=="true" || value_=="T" || value_=="True" ||
            value_=="1" || value_=="on" || value_=="On")
           d = true;
         else if(value_=="false" || value_=="F" || value_=="False" ||
                 value_=="0" || value_=="off" || value_=="Off")
           d = false;
         else 
         {
            throw edm::Exception(errors::Configuration) << name()
               << " has a bad value for bool:" << value_
               << "\nfrom " << traceback();
         }
         return Entry(name(), d, tracked_);
     }
     else {
         throw edm::Exception(errors::Configuration)
           << "Bad Entry Node type: " << type()
           << "\nfrom " << traceback();
     }

   }

     void EntryNode::checkParse(const std::string & s, char * end) const
     {
       if(*end != 0)
       {
         throw edm::Exception(errors::Configuration) <<  "Cannot create a value of type " << type()
            <<  " for parameter " << name() << " from input " << s
            << "\nIncluded from " << traceback();
       }
     }
        

  }
}

