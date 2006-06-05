#include "FWCore/ParameterSet/interface/IncludeNode.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/Visitor.h"

using std::string;

namespace edm {
  namespace pset {

    //make an empty CompositeNode
    IncludeNode::IncludeNode(const string & type, const string & name, int line)
    : CompositeNode(name, NodePtrListPtr(new NodePtrList), line),
      type_(type)
    {
    }


    void IncludeNode::accept(Visitor& v) const 
    {
      v.visitInclude(*this);
    }


    void IncludeNode::resolve(std::list<string> & openFiles)
    {
      // we don't allow circular opening of already-open files,
      // and we ignore second includes of files
      checkCircularInclusion(openFiles);
      openFiles.push_back(name);

      // make sure the openFiles list isn't corrupted
      assert(openFiles.back() == name);
      openFiles.pop_back();
    }


    void IncludeNode::checkCircularInclusion(const std::list<string> & openFiles) const
    {
      std::list<string>::const_iterator nameItr
        = std::find(openFiles.begin(), openFiles.end(), name);
      if(nameItr != openFiles.end())
      {
        throw edm::Exception(errors::Configuration, "IncludeError")
         << "Circular inclusion of file " << name;
      }
    }


  }
}

