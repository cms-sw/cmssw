#include "FWCore/ParameterSet/interface/RenamedIncludeNode.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iosfwd>

namespace edm {
  namespace pset {

    RenamedIncludeNode::RenamedIncludeNode(const std::string & type, const std::string & name,
                         const std::string & targetType, const std::string & newName, 
                         const std::string & targetName, int line)
    : IncludeNode("includeRenamed", name, line),
      targetType_(targetType),
      newName_(newName),
      targetName_(targetName)
    {
    }


    bool RenamedIncludeNode::checkTarget(NodePtr node) const
    {
      bool result = ( node->name() == targetName_ );
      if( result )
      {
        // check that the type was specified correctly
        if(node->type() != targetType_)
        {
          throw edm::Exception(errors::Configuration)
            << "Included node " << newName_
            << "is of type " << node->type()
            << ", not " <<  targetType_
            << "\nfrom " << traceback();
        }
        node->setName(newName_);
        node->setCloned(true);
      }
      // always true, but might have to rename a node
      return result;
    }


    bool RenamedIncludeNode::check(bool strict) const
    {
      //@@ doesn't check for things like multiple modules in a file
      bool found = false;
      for(NodePtrList::iterator nodeItr = nodes_->begin(), nodeItrEnd = nodes_->end();
          !found && nodeItr != nodeItrEnd; ++nodeItr)
      {
        found = checkTarget(*nodeItr);
      }

      if(!found)
      {
        throw edm::Exception(errors::Configuration)
          << "Could not find node " << targetName_
          << " in file " << name()
          << "\nfrom " << traceback();
      }
      return found;
    }


    void RenamedIncludeNode::print(std::ostream & out, 
                                   Node::PrintOptions options) const
    {
      if(options == COMPRESSED  && !isModified())
      {
        out << targetType_ << " " << newName_ << " = " << targetName_ 
            << " from \"" << name() << "\"\n";
      }
      else
      {
        // expand
        IncludeNode::print(out, options);
      }
    }

  }
}

