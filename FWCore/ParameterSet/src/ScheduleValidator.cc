/**
   \file
   Implementation of class ScheduleValidator

   \author Stefano ARGIRO
   \version $Id: ScheduleValidator.cc,v 1.17 2007/05/15 22:58:50 rpw Exp $
   \date 10 Jun 2005
*/

static const char CVSId[] = "$Id: ScheduleValidator.cc,v 1.17 2007/05/15 22:58:50 rpw Exp $";

#include "FWCore/ParameterSet/src/ScheduleValidator.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <sstream>
#include <iterator>
#include <iostream>
using namespace edm;
using namespace edm::pset;
using namespace std;

 
ScheduleValidator::ScheduleValidator(const ScheduleValidator::PathContainer& 
				     pathFragments,
				     const ParameterSet& processPSet): 
  nodes_(pathFragments),
  processPSet_(processPSet),
  leaves_(),
  dependencies_() 
{
 
  vector<string> paths = processPSet.getParameter<vector<string> >("@paths");

  for(vector<string>::const_iterator pathIt = paths.begin(), pathItEnd = paths.end();
      pathIt != pathItEnd; 
      ++pathIt) {
    NodePtr head = findPathHead(*pathIt);
    gatherLeafNodes(head);
  }//for

}


NodePtr ScheduleValidator::findPathHead(string pathName){
  // cout << "in findPathHead" << endl;
  for (PathContainer::iterator pathIt = nodes_.begin(), pathItEnd = nodes_.end();
       pathIt != pathItEnd; ++pathIt) {
//cout << "  looking at " << (*pathIt)->type() << " " << (*pathIt)->name << endl;
    if ((*pathIt)->type() != "path" &&
        (*pathIt)->type() != "endpath") continue;
    if ((*pathIt)->name() == pathName) return ((*pathIt)->wrapped());

  }// for
  throw edm::Exception(errors::Configuration) << "Cannot find a path named " << pathName;
  NodePtr ret;
  return ret;
}

void ScheduleValidator::gatherLeafNodes(NodePtr& basenode){

  if (basenode->type() == "," || basenode->type() == "&"){
    OperatorNode* onode = dynamic_cast<OperatorNode*>(basenode.get());
    gatherLeafNodes(onode->left());
    gatherLeafNodes(onode->right());

  } else {
    leaves_.push_back(basenode);
  }

}// gatherLeafNodes




void ScheduleValidator::validate(){

  dependencies_.clear();

  std::string leafName;
  std::string sonName;
  std::string leftName;

  leafName.reserve(100);
  sonName.reserve(100);
  leftName.reserve(100);

  // iterate on leaf nodes
  for (std::list<NodePtr>::iterator leafIt = leaves_.begin(), leafItEnd = leaves_.end();
        leafIt != leafItEnd; ++leafIt) {
    
    DependencyList dep;

    // follow the tree up thru parent nodes  
    Node* p = (*leafIt)->getParent();
    Node* son = (*leafIt).get();
    
    // make sure we don't redescend right of our parent
    
    while  (p){
      // if we got operator '&' continue
      if (p->type() == "&") {
	son = p;
	p = p->getParent(); 
	continue;
      }

      OperatorNode* node = dynamic_cast<OperatorNode*>(p);
      // make sure we don't redescend where we came from

      sonName = son->name();
      if (sonName.size() > 0 &&
          (sonName[0] == '!' || sonName[0] == '-')) sonName.erase(0,1);

      leftName = node->left()->name();
      if (leftName.size() > 0 &&
          (leftName[0] == '!' || leftName[0] == '-')) leftName.erase(0,1);

      if (sonName != leftName) findDeps(node->left(), dep);
      
      son = p;
      p = p->getParent();
    } // while
        
    dep.sort();   
    dep.unique(); // removes duplicates

    // insert the list of deps

    leafName = (*leafIt)->name();
    if (leafName.length() > 0 && 
        (leafName[0] == '!' || leafName[0] == '-')) leafName.erase(0,1);

  //  validateDependencies(leafName, *leafIt,  dep);
    mergeDependencies(leafName, dep);

  }//for leaf

  validatePaths();
     
}// validate


void ScheduleValidator::validateDependencies(const std::string & leafName, const NodePtr & leafNode, const DependencyList& dep)
{
  Dependencies::iterator depIt = dependencies_.find(leafName);
std::cout << leafName << " " << std::endl;
  if (depIt != dependencies_.end()) {
    DependencyList& old_deplist = (*depIt).second;
    // if the list is different from an existing one
    // then we have an inconsitency
    if (old_deplist != dep) {

      ostringstream olddepstr,newdepstr, traceback;
      copy(old_deplist.begin(), old_deplist.end(),
            ostream_iterator<string>(olddepstr,","));
      copy(dep.begin(), dep.end(),
            ostream_iterator<string>(newdepstr,","));
      std::string olddeps = olddepstr.str();
      if(olddeps == "") olddeps = "<NOTHING>";
      std::string newdeps = newdepstr.str();
      if(newdeps == "") newdeps = "<NOTHING>";

      leafNode->printTrace(traceback);
      std::string traceStr = traceback.str();
      if(traceStr == "") traceStr = "<MAIN CFG>";

      throw edm::Exception(errors::Configuration,"InconsistentSchedule")
        << "Inconsistent schedule for module "
        << leafName
        << "\n"
        << "Depends on " << olddeps
        << " but also on " << newdeps
        << "\n"
        << "Second set of dependencies comes from: " << traceStr << "\n";
    }
  }
  else {
    dependencies_[leafName] = dep;
  }

}


void ScheduleValidator::mergeDependencies(const std::string & leafName, DependencyList& deps)
{
  dependencies_[leafName].merge(deps);
  dependencies_[leafName].unique();
}


void ScheduleValidator::validatePaths()
{
 vector<string> paths = processPSet_.getParameter<vector<string> >("@paths");
  for(vector<string>::const_iterator pathItr = paths.begin(); pathItr != paths.end(); ++pathItr)
  {
    validatePath(*pathItr);
  }
}


void ScheduleValidator::validatePath(const std::string & path) 
{
  std::vector<std::string> schedule = processPSet_.getParameter<vector<string> >(path);
  std::vector<std::string>::iterator module = schedule.begin(),
    lastModule = schedule.end();
  for( ; module != lastModule; ++module)
  {
     Dependencies::iterator depList = dependencies_.find(*module);
     if(depList == dependencies_.end())
     {
        throw edm::Exception(errors::Configuration,"InconsistentSchedule")
         << "No dependecies calculated for " << *module;
     }
     else 
     {
       DependencyList::iterator depItr = depList->second.begin(),
          lastDep = depList->second.end();
       for( ; depItr != lastDep; ++depItr)
       {
         // make sure each dependency is in the schedule before module
         if(std::find(schedule.begin(), module, *depItr) == module)
         {
           ostringstream pathdump;
           copy(schedule.begin(), schedule.end(),
             ostream_iterator<string>(pathdump," "));
           
           throw edm::Exception(errors::Configuration,"InconsistentSchedule")
          << "Module " << *module << " depends on " << *depItr
          << "\n"
          << " but path " << path << "  contains "  << pathdump.str()
          << "\n";
         }
       }
     }
  }
}


void ScheduleValidator::findDeps(NodePtr& node, DependencyList& dep){

  // if we have an operand, add it to the list of dependencies
  if (node->type() == "operand") {

    if (node->name().size() > 0 &&
        (node->name()[0] == '!' || node->name()[0] == '-')) {
      std::string nodeName = node->name();
      nodeName.erase(0,1);
      dep.push_back(nodeName);
    }
    else dep.push_back(node->name());
  }
  // else follow the tree, unless the leaf is contained in the node
  else{

    OperatorNode* opnode = dynamic_cast<OperatorNode*>(node.get());
    
    findDeps(opnode->left(),dep);
    findDeps(opnode->right(),dep);
	   
  }
}// findDeps


std::string 
ScheduleValidator::dependencies(const std::string& modulename) const{


  Dependencies::const_iterator depIt = dependencies_.find(modulename);
  if (depIt == dependencies_.end()){
    ostringstream err;
    throw edm::Exception(errors::Configuration,"ScheduleDependencies")
      << "Error : Dependecies for "
      << modulename << " were not calculated";
  }

  ostringstream deplist;
  copy((*depIt).second.begin(), (*depIt).second.end(), 
	      ostream_iterator<string>(deplist,","));
  return deplist.str();

}
