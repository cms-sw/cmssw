/**
   \file
   Implementation of class ScheduleValidator

   \author Stefano ARGIRO
   \version $Id: ScheduleValidator.cc,v 1.14 2007/01/20 00:09:56 wmtan Exp $
   \date 10 Jun 2005
*/

static const char CVSId[] = "$Id: ScheduleValidator.cc,v 1.14 2007/01/20 00:09:56 wmtan Exp $";

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

    Dependencies::iterator depIt = dependencies_.find(leafName);
    if (depIt != dependencies_.end()) {
      DependencyList& old_deplist = (*depIt).second;
    
      // if the list is different from an existing one
      // then we have an inconsitency
      if (old_deplist != dep) {

	ostringstream olddepstr,newdepstr;
	copy(old_deplist.begin(), old_deplist.end(), 
	      ostream_iterator<string>(olddepstr,","));
	copy(dep.begin(), dep.end(), 
	      ostream_iterator<string>(newdepstr,","));

	throw edm::Exception(errors::Configuration,"InconsistentSchedule")
	  << "Inconsistent schedule for module "
	  << leafName
	  << "\n"
	  << "Depends on " << olddepstr.str() 
	  << " but also on " << newdepstr.str()
	  << "\n";
      }
    }
    else {
      dependencies_[leafName] = dep;
    }

  }//for leaf
     
}// validate

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
