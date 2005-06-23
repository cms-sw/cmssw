
/**
   \file
   Declaration of class ProcessPSetBuilder

   \author Stefano ARGIRO
   \version $Id: ProcessPSetBuilder.h,v 1.1 2005/06/20 15:22:03 argiro Exp $
   \date 16 Jun 2005
*/

#ifndef _edm_ProcessPSetBuilder_h_
#define _edm_ProcessPSetBuilder_h_

static const char CVSId_edm_ProcessPSetBuilder[] = 
"$Id: ProcessPSetBuilder.h,v 1.1 2005/06/20 15:22:03 argiro Exp $";

#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>
#include <map>

namespace edm {

  namespace pset{class Node;class WrapperNode;}
  class ScheduleValidator;
  class ParameterSet;
  class ProcessDesc;
    /**
     \class ProcessPSetBuilder ProcessPSetBuilder.h "edm/ProcessPSetBuilder.h"

     \brief Encapsulates the functionality to build a Process ParameterSet
            from the configuration language

     \author Stefano ARGIRO
     \date 16 Jun 2005
  */
  class ProcessPSetBuilder {

  public:
    /// construct from the configuration language string
    ProcessPSetBuilder(const std::string& config);

    ~ProcessPSetBuilder();
 
    /// get the ParameterSet that describes the process
    boost::shared_ptr<ParameterSet> getProcessPSet() const;
    
    /// get the dependencies for this module
    /** the return string is a list of comma-separated 
      * names of the modules on which modulename depends*/
    std::string  getDependencies(const std::string& modulename);

  private:
    
    typedef boost::shared_ptr<pset::WrapperNode> WrapperNodePtr;
    typedef std::vector<std::string> Strs;
    typedef std::map<std::string, WrapperNodePtr > SeqMap;
    typedef boost::shared_ptr<pset::Node> NodePtr;

    /// recursively extract names of modules and store them in Strs;
    void getNames(const pset::Node* n, Strs& out);

    /// perform sequence substitution for this node
    void sequenceSubstitution(NodePtr& node, SeqMap&  sequences);

    /// Take a path Wrapper node and extract names
    /** put the name of this path in @param paths
     *  and put the names of the modules for this path in @param out */
    void fillPath(WrapperNodePtr n, Strs& paths, ParameterSet* out);

    /// diagnostic function
    void dumpTree(NodePtr& node);

    /// the validation object
    ScheduleValidator*  validator_;
    
    boost::shared_ptr<ProcessDesc>   processDesc_;

    boost::shared_ptr<ParameterSet> processPSet_;
  };
} // edm


#endif // _edm_ProcessPSetBuilder_h_





