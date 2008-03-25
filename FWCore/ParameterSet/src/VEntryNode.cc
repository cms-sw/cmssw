#include "FWCore/ParameterSet/interface/VEntryNode.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/ParameterSet/interface/EntryNode.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <boost/cstdint.hpp>
#include <ostream>
#include <iterator>

using std::string;
using std::vector;

namespace edm {
  namespace pset {


    VEntryNode::VEntryNode(const string& t, const string& n,
                           StringListPtr v,bool untracked, int line):
      Node(n, line),
      type_(t),
      value_(v),
      tracked_(!untracked)
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
      const char* t = tracked_ ? "" : "untracked ";
      ost << t << type() << " " << name() << " = {\n  ";

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

    void VEntryNode::locate(const std::string & s, std::ostream & out) const
    {
      // look in the vector for a matching substring
      std::string match = "";
      for(StringList::const_iterator i = value_->begin(), iEnd = value_->end(); i != iEnd; ++i)
      {
        if((*i).find(s,0) != std::string::npos)
        {
          match = *i;
        }
      }

      // now try the name of the node itself
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


    void VEntryNode::accept(Visitor& v) const
    {
      v.visitVEntry(*this);
    }


    void VEntryNode::replaceWith(const ReplaceNode * replaceNode)
    {
      // see if it's a replace or an append
      if(replaceNode->type() == "replace")
      {
        VEntryNode * replacement = replaceNode->value<VEntryNode>();
        if(replacement == 0) {
          throw edm::Exception(errors::Configuration)
            << "Cannot replace entry vector" << name()
            <<   " with " << replaceNode->type()
            << "\nfrom " << traceback();
        }
        // replace the value, keep the type
        value_ = replacement->value_;
      }
      else if(replaceNode->type() == "replaceAppend")
      {
        append(replaceNode->value());
      }
      else 
      {
         throw edm::Exception(errors::Configuration)
            << "Cannot replace entry vector" << name()
            <<   " with " << replaceNode->type()
            << "\nfrom " << traceback();
      }
      setModified(true);
    }


    void VEntryNode::append(NodePtr ptr)
    {  
      // single or multiple?
      EntryNode * entryNode =  dynamic_cast<EntryNode*>(ptr.get());
      if(entryNode != 0)
      {
        value_->push_back(entryNode->value());
      }
      else 
      {
        // try VEntry
        VEntryNode * ventryNode =  dynamic_cast<VEntryNode*>(ptr.get());
        if(ventryNode != 0)
        {
          StringListPtr entries = ventryNode->value_;
          for(StringList::const_iterator itr = entries->begin(), itrEnd = entries->end();
              itr != itrEnd; ++itr)
          {
            value_->push_back(*itr);
          }
        }
        // neither Entry or VEntry
        else 
        { 
          throw edm::Exception(errors::Configuration)
            << "Bad type to append to VEntry " 
            <<  ptr->type()
            << " \nfrom " << traceback();
        }
      }
    }  
    
    edm::Entry VEntryNode::makeEntry() const
    {
      vector<string>::const_iterator ib(value_->begin()),
        ie(value_->end()),k=ib;

     if(type()=="vstring")
       {
         vector<string> usethis;
         for(;ib!=ie;++ib) usethis.push_back(withoutQuotes(*ib));
         return Entry(name(), usethis, tracked_);
       }
     else if(type()=="vdouble")
       {
         vector<double> d ;
         for(ib=k;ib!=ie;++ib) d.push_back(strtod(ib->c_str(),0));
         return Entry(name(), d, tracked_);
       }
     else if(type()=="vint32")
       {
         vector<int> d ;
         for(ib=k;ib!=ie;++ib) d.push_back(atoi(ib->c_str()));
         return Entry(name(), d, tracked_);
       }
     else if(type()=="vuint32")
       {
         vector<unsigned int> d ;
         for(ib=k;ib!=ie;++ib) d.push_back(strtoul(ib->c_str(),0,10));
         return Entry(name(), d, tracked_);
       }
     else if(type()=="vint64")
       {
         vector<boost::int64_t> d ;
         for(ib=k;ib!=ie;++ib) d.push_back(atoi(ib->c_str()));
         return Entry(name(), d, tracked_);
       }
     else if(type()=="vuint64")
       {
         vector<boost::uint64_t> d ;
         for(ib=k;ib!=ie;++ib) d.push_back(strtoul(ib->c_str(),0,10));
         return Entry(name(), d, tracked_);
       }
     else if(type()=="VInputTag")
       {
         vector<InputTag> d ;
         for(ib=k;ib!=ie;++ib) d.push_back( InputTag(withoutQuotes(*ib)) );
         return Entry(name(), d, tracked_);
       }
     else if(type()=="VEventID")
       {
         vector<edm::EventID> d ;
         for(ib=k;ib!=ie;++ib)
         {
           edm::EventID eventID;
           edm::decode(eventID, *ib);
           d.push_back( eventID );
         }
         return Entry(name(), d, tracked_);
       }
     else
       {
         throw edm::Exception(errors::Configuration)
           << "Bad VEntry Node type: " << type()
           << "\nfrom " << traceback();
       }
    }

  }
}

