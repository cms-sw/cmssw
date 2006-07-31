#include "FWCore/ParameterSet/interface/IncludeNode.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <iostream>
#include <iterator>
using std::string;

namespace edm {
  namespace pset {

    //make an empty CompositeNode
    IncludeNode::IncludeNode(const string & type, const string & name, int line)
    : CompositeNode(withoutQuotes(name), NodePtrListPtr(new NodePtrList), line),
      type_(type)
    {
    }


    void IncludeNode::accept(Visitor& v) const 
    {
      v.visitInclude(*this);
    }


    void IncludeNode::print(std::ostream & ost, Node::PrintOptions options) const
    {
      // if it's modified, we have to print out everything
      if(options == COMPRESSED && !isModified())
      {
         ost << "include \"" << name << "\"\n";
      }
      else 
      {
        // we can't just take CompositeNode's print, since we don't want the 
        // curly braces around everything
        NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
        for(;i!=e;++i)
        {
          (**i).print(ost, options);
          ost << "\n";
        } 
      }
    }

   
    void IncludeNode::printTrace(std::ostream & ost) const
    {
      if(isResolved())
      {
        ost << "Line " << line << " includes " << fullPath_ << "\n";
      }
      // and pass it up
      Node::printTrace(ost);
    }


    void IncludeNode::resolve(std::list<string> & openFiles,
                              std::list<string> & sameLevelIncludes)
    {
      // we don't allow circular opening of already-open files,
      if(std::find(openFiles.begin(), openFiles.end(), name)
         != openFiles.end())
      {
        throw edm::Exception(errors::Configuration, "IncludeError")
         << "Circular inclusion of file " << name;
      }

      // ignore second includes at the same level
      std::list<std::string>::const_iterator twinSister
        = find(sameLevelIncludes.begin(), sameLevelIncludes.end(), name);
      if(twinSister != sameLevelIncludes.end())
      {
        // duplicate.  Remove this one.
        CompositeNode * parent  = dynamic_cast<CompositeNode *>(getParent());
        assert(parent != 0);
        parent->removeChild(this);
      }
      else 
      {
        openFiles.push_back(name);
        sameLevelIncludes.push_back(name);
        FileInPath fip(name);
        fullPath_ = fip.fullPath();
        isResolved_ = true;
        string configuration;
        read_whole_file(fip.fullPath(), configuration);

        // save the name of the file
        extern string currentFile;
        string oldFile = currentFile;
        currentFile = fullPath_;
        nodes_ = parse(configuration.c_str());
        // put in the backwards links right away
        setAsChildrensParent();
        // resolve the includes in any subnodes
        CompositeNode::resolve(openFiles, sameLevelIncludes);
      
        currentFile = oldFile;
        // make sure the openFiles list isn't corrupted
        assert(openFiles.back() == name);
        openFiles.pop_back();
      }
    }


    void IncludeNode::insertInto(edm::ProcessDesc & procDesc) const
    {
      // maybe refactor this down to CompositeNode, if another
      // CompositeNode needs it
      NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
      for(;i!=e;++i)
      {
        (**i).insertInto(procDesc);
      }
    }


  }
}

