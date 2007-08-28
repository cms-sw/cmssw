#include "FWCore/ParameterSet/interface/IncludeNode.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
// circular dependence here
#include "FWCore/ParameterSet/interface/ModuleNode.h"

namespace edm {
  namespace pset {

    //make an empty CompositeNode
    IncludeNode::IncludeNode(const std::string & type, const std::string & name, int line)
    : CompositeNode(withoutQuotes(name), NodePtrListPtr(new NodePtrList), line),
      type_(type),   
      fullPath_(""),
      isResolved_(false)
    {
    }


    void IncludeNode::accept(Visitor& v) const 
    {
      v.visitInclude(*this);
    }


    void IncludeNode::dotDelimitedPath(std::string & path) const
    {
      // don't add your name
      Node * parent = getParent();
      if(parent != 0)
      {
        parent->dotDelimitedPath(path);
      }
    }


    void IncludeNode::print(std::ostream & ost, Node::PrintOptions options) const
    {
      // if it's modified, we have to print out everything
      if(options == COMPRESSED && !isModified())
      {
         ost << "include \"" << name() << "\"\n";
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
        ost << "Line " << line() << " includes " << fullPath_ << "\n";
      }
      // and pass it up
      Node::printTrace(ost);
    }


    void IncludeNode::resolve(std::list<std::string> & openFiles,
                              std::list<std::string> & sameLevelIncludes,
                              bool strict)
    {
      // we don't allow circular opening of already-open files,
      if(std::find(openFiles.begin(), openFiles.end(), name())
         != openFiles.end())
      {
        throw edm::Exception(errors::Configuration, "IncludeError")
         << "Circular inclusion of file " << name()
         << "\nfrom " << traceback();
      }

      // ignore second includes at the same level
      bool ignore = false;
      if(checkMultipleIncludes())
      {
        std::list<std::string>::const_iterator twinSister
          = std::find(sameLevelIncludes.begin(), sameLevelIncludes.end(), name());
        if(twinSister != sameLevelIncludes.end())
        {
          // duplicate.  Remove this one.
          CompositeNode * parent  = dynamic_cast<CompositeNode *>(getParent());
          assert(parent != 0);
          parent->removeChild(this);
          ignore = true;
        }
        else
        {
          sameLevelIncludes.push_back(name());
        }
      }

      if(!ignore)
      {
        openFiles.push_back(name());
        try
        {
          FileInPath fip(name());
          fullPath_ = fip.fullPath();
        }
        catch(const edm::Exception & e)
        {
          // re-throw with the traceback info
          throw edm::Exception(errors::Configuration, "IncludeError")
          << "Exception found trying to include " << name() << ":\n"
          << e.what() << "\nIncluded from:\n" << traceback();
        }

        isResolved_ = true;
        std::string configuration = read_whole_file(fullPath_);
        // save the name of the file
        extern std::string currentFile;
        std::string oldFile = currentFile;
        currentFile = fullPath_;
        nodes_ = parse(configuration.c_str());
        // put in the backwards links right away
        setAsChildrensParent();
        // resolve the includes in any subnodes
        CompositeNode::resolve(openFiles, sameLevelIncludes, strict);
      
        currentFile = oldFile;
        // make sure the openFiles list isn't corrupted
        assert(openFiles.back() == name());
        openFiles.pop_back();
        check(strict);
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


    bool IncludeNode::check(bool strict) const
    {
      bool ok = true;
      int nletters = name().length();
      assert(nletters >= 3);
      std::string lastThreeLetters = name().substr(nletters-3);
      // count the number of module nodes
      int nModules = 0;

      if(lastThreeLetters == "cfi")
      {
        NodePtrList::const_iterator i(nodes_->begin()),e(nodes_->end());
        for(;i!=e;++i)
        {
          if(dynamic_cast<const ModuleNode *>((*i).get()) != 0)
          {
            ++nModules;
          }
        }
       
        if(nModules > 1)
        {
          ok = false;
          std::ostringstream message;
   
          message
           << nModules << " modules were defined in " 
           << name() << ".\nOnly one module should be defined per .cfi."
           <<"\nfrom: " << traceback();

          if(strict)
          {
            throw edm::Exception(errors::Configuration) << message.str();
          }

        }
      }

      // now check if this is included from a .cfi
      if(includeParentSuffix() == "cfi")
      {
        if(strict) 
        {
          // the only object that can be included within a .cfi is a block.
          // see if this include file has only one node, a block.
          // tolerate further nesting for now, as long as the result
          // is just one block
          NodePtrListPtr kids = children();
          ok = (kids->size()==1 && kids->front()->type() == "block");

          if(!ok)
          {
            std::ostringstream message;
            message << "include statements should not be used in a .cfi file."
                    << "\nfrom:" << traceback();
            throw edm::Exception(errors::Configuration) << message.str();
          }
        }
      }

      return ok;
    }


  }
}

