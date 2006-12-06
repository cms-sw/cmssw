#include "FWCore/ParameterSet/interface/RenamedIncludeNode.h"
#include "FWCore/ParameterSet/interface/IncludeFileFinder.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>

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


    bool RenamedIncludeNode::okToInclude(NodePtr node) const
    {
      // only take the node specified
      bool result = (node->name() == targetName_);
      if(result)
      {
        // check that the type was specified correctly
        if(node->type() != targetType_)
        {
          throw edm::Exception(errors::Configuration)
            << "Included node " << newName_
            << "is of type " << node->type()
            << ", not " <<  targetType_;
        }
     
        node->setName(newName_);
        node->setCloned(true);
      }
      return result;
    }


    void RenamedIncludeNode::resolve(std::list<std::string> & openFiles,
                             std::list<std::string> & sameLevelIncludes)
    {
      IncludeNode::resolve(openFiles, sameLevelIncludes);
      filterNodes();
      if(nodes_->size() != 1)
      {
        throw edm::Exception(errors::Configuration)
          << "Could not find node " << targetName_ 
          << " in file " << name();
      }
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

