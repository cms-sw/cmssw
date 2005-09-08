/**
   \file
   Implementation of calss ProcessPSetBuilder

   \author Stefano ARGIRO
   \version $Id: ProcessPSetBuilder.cc,v 1.3 2005/07/14 16:17:23 jbk Exp $
   \date 17 Jun 2005
*/

static const char CVSId[] = "$Id: ProcessPSetBuilder.cc,v 1.3 2005/07/14 16:17:23 jbk Exp $";


#include <FWCore/ParameterSet/interface/ProcessPSetBuilder.h>
#include "FWCore/ParameterSet/interface/Makers.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/Nodes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/Entry.h"

#include "FWCore/ParameterSet/src/ScheduleValidator.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "boost/shared_ptr.hpp"
#include <vector>
#include <map>
#include <stdexcept>
#include <iostream>

using namespace boost;
using namespace edm;
using namespace edm::pset;
using namespace std;

ProcessPSetBuilder::~ProcessPSetBuilder(){delete validator_;}

ProcessPSetBuilder::ProcessPSetBuilder(const std::string& config){
  
  boost::shared_ptr<edm::pset::NodePtrList> nodelist = 
    edm::pset::parse(config.c_str());
  if(0 == nodelist.get()) {
    throw edm::Exception(errors::Configuration,"FileOpen")
      << "Unable to parse configuration file.\n"
      << "Please check the error message reported earlier.";
  }
  
  processDesc_= edm::pset::makeProcess(nodelist);

   SeqMap sequences;

   // loop on path fragments
   ProcessDesc::PathContainer::iterator pathIt; 
   Strs pathnames;
   
   for(pathIt= processDesc_->pathFragments_.begin();
       pathIt!=processDesc_->pathFragments_.end();
       ++pathIt){
     
     if ((*pathIt)->type()=="sequence") {
       sequences[(*pathIt)->name()]= (*pathIt);
     }
     
     if ((*pathIt)->type()=="path") {
       sequenceSubstitution((*pathIt)->wrapped_, sequences);
       fillPath((*pathIt),pathnames,&processDesc_->pset_);
     }

     if ((*pathIt)->type()=="endpath") {
	//cout << "got endpath = " << (*pathIt)->name() << endl;
	//cout << "pointer = " << typeid(*(*pathIt)->wrapped_.get()).name() << endl;
       sequenceSubstitution((*pathIt)->wrapped_, sequences);
       fillPath((*pathIt),pathnames,&processDesc_->pset_);
     }
     
     
   } // loop on path fragments
   
   processDesc_->pset_.insert(true,"paths",Entry(pathnames,true));
   
   validator_= 
     new ScheduleValidator(processDesc_->pathFragments_,processDesc_->pset_); 
   
   validator_->validate();
   
   processPSet_= 
     shared_ptr<edm::ParameterSet>(new ParameterSet(processDesc_->pset_));
}


void ProcessPSetBuilder::getNames(const Node* n, Strs& out){
  if(n->type()=="operand"){ 
    out.push_back(n->name());
  } else {	
    const OperatorNode& op = dynamic_cast<const OperatorNode&>(*n);
    getNames(op.left_.get(),out);
    getNames(op.right_.get(),out);
  }
} // getNames


void ProcessPSetBuilder::fillPath(WrapperNodePtr n, Strs&   paths,  
			      ParameterSet* out){
  
  Strs names;
  getNames(n->wrapped_.get(),names);    
  out->insert(true,n->name(),Entry(names,true));
  paths.push_back(n->name()); // add to the list of paths
  
} // fillPath(..) 



void ProcessPSetBuilder::sequenceSubstitution(NodePtr& node, 
			    SeqMap&  sequences){
  
  if (node->type()=="operand"){
    SeqMap::iterator seqIt = sequences.find(node->name()); 
    if (seqIt!= sequences.end()){
      node = seqIt->second->wrapped_;
    }
  } // if operator
  else {
    OperatorNode* onode = dynamic_cast<OperatorNode*>(node.get());
    
    
    SeqMap::iterator seqIt = sequences.find(onode->left_->name()); 
    if (seqIt!= sequences.end()) {
      onode->left_= seqIt->second->wrapped_;
      onode->left_->setParent(onode);
    }
    seqIt = sequences.find(onode->right_->name()); 
    if (seqIt!= sequences.end()){
      onode->right_= seqIt->second->wrapped_; 
      onode->right_->setParent(onode);
    }
    sequenceSubstitution(onode->left_, sequences);
    sequenceSubstitution(onode->right_,sequences);
    
  }// else (operand)
  
} // sequenceSubstitution


void ProcessPSetBuilder::dumpTree(NodePtr& node){
  if(node->type()=="operand"){ 
    cout << " Operand " << node->name()<< " p:";
    if (node->getParent()) cout <<  node->getParent()->name();cout<< endl;
  } else{	
    OperatorNode* op = dynamic_cast<OperatorNode*>(node.get());
    cout << " Operator: " << op->name()<<"["<<op->type()<<"]" 
	 << " l:" << op->left_ << " r:"<<op->right_<< " p:";
    if (op->parent_)cout<<  op->parent_->name();cout<< endl;
    dumpTree(op->left_);
    dumpTree(op->right_);
  }
} // dumpTree


std::string  ProcessPSetBuilder::getDependencies(const std::string& modulename){
  return validator_->dependencies(modulename);
}


boost::shared_ptr<edm::ParameterSet>  
ProcessPSetBuilder::getProcessPSet() const{

  return processPSet_;

}
