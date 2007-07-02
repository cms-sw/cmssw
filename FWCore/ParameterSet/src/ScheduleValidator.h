#ifndef ParameterSet_ScheduleValidator_h
#define ParameterSet_ScheduleValidator_h

/**
   \file
   Declaration of class ScheduleValidator

   \author Stefano ARGIRO
   \version $Id: ScheduleValidator.h,v 1.4 2007/05/22 21:47:13 rpw Exp $
   \date 10 Jun 2005
*/

#include <string>
#include <list>
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "FWCore/ParameterSet/interface/Nodes.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>

namespace edm {

  class ParameterSet;
  /**
     \class ScheduleValidator ScheduleValidator.h "edm/ScheduleValidator.h"

     \brief Incapsulates the machinery to validate a schedule

     \author Stefano ARGIRO
     \date 10 Jun 2005
  */
  class ScheduleValidator {
  public:

    typedef boost::shared_ptr<pset::WrapperNode> WrapperNodePtr ;
    typedef std::vector< WrapperNodePtr > PathContainer;
    typedef std::list<std::string> DependencyList;
    typedef std::map<std::string, DependencyList> Dependencies;

    /// construct a validator object
    /** Needs the tree from ProcessDesc (@param path) and
        the reduced path, that is the path after sequence substitution
        @param reduced_path */ 
    ScheduleValidator(const PathContainer& path, 
		      const ParameterSet&  processPSet);

    /// Diagnostic function that returns the dependency list for  modulename
    /** Throws if deps were not calculated for module modulename*/
    std::string dependencies(const std::string& modulename) const;

    /// validates the schedule, throws in case of inconsitency
    void validate();

  private:
    /// if the module name begins with '!' or '-', erase the character
    void removeUnaries(std::string & moduleName);

    /// fill the list of leaves of the basenode
    void gatherLeafNodes(pset::NodePtr& basenode);
   
    /// find the node that is the root for path pathName
    edm::pset::NodePtr findPathHead(std::string pathName);
   
    /// find dependencies (module names) from this node down
    void  findDeps(edm::pset::NodePtr& node, DependencyList& dep);

    void validateDependencies(const std::string & leafName, 
                              const pset::NodePtr & leafNode, const DependencyList& deps);
    void mergeDependencies(const std::string & leafName, DependencyList& deps);
    void validatePaths();

    /// checks the path to see if the DependencyList is satisfied
    void validatePath(const std::string & path);


    /// The tree
    PathContainer                      nodes_; 
    /// process Pset
    ParameterSet                       processPSet_;
 
    /// list of leaf nodes
    std::list<pset::NodePtr>           leaves_;
    /// maps module name to list of its dependencies
    Dependencies                       dependencies_;
 
  }; // ScheduleValidator


} // edm


#endif
