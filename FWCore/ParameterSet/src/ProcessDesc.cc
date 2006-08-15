/**
   \file
   Implementation of calss ProcessDesc

   \author Stefano ARGIRO
   \version $Id: ProcessDesc.cc,v 1.5 2006/08/07 23:52:36 rpw Exp $
   \date 17 Jun 2005
*/

static const char CVSId[] = "$Id: ProcessDesc.cc,v 1.5 2006/08/07 23:52:36 rpw Exp $";


#include <FWCore/ParameterSet/interface/ProcessDesc.h>
#include "FWCore/ParameterSet/interface/Makers.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Entry.h"

#include "FWCore/ParameterSet/src/ScheduleValidator.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include "FWCore/ParameterSet/interface/PSetNode.h"

#include "boost/shared_ptr.hpp"
#include <vector>
#include <map>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <iterator>

using namespace boost;
using namespace std;

namespace edm
{

  ProcessDesc::~ProcessDesc()
  {
    delete validator_;
  }

  ProcessDesc::ProcessDesc(const std::string& config)
  : pset_(new ParameterSet),
    services_(new std::vector<ParameterSet>())
  {
    edm::pset::ParseResults parsetree = edm::pset::fullParse(config.c_str());

    // top node should be the PSetNode representing the process
    pset::NodePtr processPSetNodePtr = parsetree->front();
    edm::pset::PSetNode * processPSetNode 
      = dynamic_cast<edm::pset::PSetNode*>(processPSetNodePtr.get());
    assert(processPSetNode != 0);
    processPSetNode->fillProcess(*this);


    writeBookkeeping("@all_modules");
    writeBookkeeping("@all_sources");
    writeBookkeeping("@all_loopers");
    writeBookkeeping("@all_esmodules");
    writeBookkeeping("@all_essources");
    writeBookkeeping("@all_esprefers");

    SeqMap sequences;

    // loop on path fragments
    ProcessDesc::PathContainer::iterator pathIt; 
    Strs endpaths, triggerpaths;
   
    for(pathIt= pathFragments_.begin();
	pathIt!=pathFragments_.end();
	++pathIt){
     
      if ((*pathIt)->type()=="sequence") {
	sequences[(*pathIt)->name()]= (*pathIt);
      }
     
      if ((*pathIt)->type()=="path") {
	sequenceSubstitution((*pathIt)->wrapped(), sequences);
	fillPath((*pathIt),triggerpaths);
      }

      if ((*pathIt)->type()=="endpath") {
	//cout << "got endpath = " << (*pathIt)->name << endl;
	//cout << "pointer = " << typeid(*(*pathIt)->wrapped().get()).name << endl;
	sequenceSubstitution((*pathIt)->wrapped(), sequences);
	fillPath((*pathIt),endpaths);
      }
     
     
    } // loop on path fragments
    
    Strs pathnames(triggerpaths);
    pathnames.insert(pathnames.end(),endpaths.begin(),endpaths.end());

    if(1 <= edm::debugit())
      {
	std::cerr << "\npathnames=\n  ";
	std::copy(pathnames.begin(),pathnames.end(),
		  std::ostream_iterator<std::string>(std::cerr,","));
	std::cerr << "\ntriggernames=\n  ";
	std::copy(triggerpaths.begin(),triggerpaths.end(),
		  std::ostream_iterator<std::string>(std::cerr,","));
	std::cerr << "\nendpaths=\n  ";
	std::copy(endpaths.begin(),endpaths.end(),
		  std::ostream_iterator<std::string>(std::cerr,","));
	std::cerr << "\n";
      }

    ParameterSet paths_trig;
    paths_trig.addParameter("@paths",triggerpaths);
    paths_trig.addParameter("@end_paths",endpaths);

    pset_->addUntrackedParameter("@trigger_paths",paths_trig);
    pset_->addParameter("@paths",pathnames);
   
    validator_= 
      new ScheduleValidator(pathFragments_,*pset_); 
   
    validator_->validate();
//std::cout << *pset_ << std::endl; 
  }


  void ProcessDesc::record(const std::string & index, const std::string & name) 
  {
    bookkeeping_[index].push_back(name);
  }

  void ProcessDesc::writeBookkeeping(const std::string & name)
  {
    pset_->addParameter(name, bookkeeping_[name]);
  }
 

  void 
  ProcessDesc::getNames(const edm::pset::Node* n, Strs& out){
    if(n->type()=="operand"){ 
      out.push_back(n->name());
    } else {	
      const edm::pset::OperatorNode& op = dynamic_cast<const edm::pset::OperatorNode&>(*n);
      getNames(op.left().get(),out);
      getNames(op.right().get(),out);
    }
  } // getNames


  void ProcessDesc::fillPath(WrapperNodePtr n, Strs&   paths)
  {
  
    Strs names;
    getNames(n->wrapped().get(),names);    
    pset_->addParameter(n->name(),names);
    paths.push_back(n->name()); // add to the list of paths
  
  } // fillPath(..) 



  void ProcessDesc::sequenceSubstitution(NodePtr& node, 
						SeqMap&  sequences){
  
    if (node->type()=="operand"){
      SeqMap::iterator seqIt = sequences.find(node->name()); 
      if (seqIt!= sequences.end()){
        node = seqIt->second->wrapped();
        sequenceSubstitution(node, sequences);
      }
    } // if operator
    else {
      edm::pset::OperatorNode* onode = dynamic_cast<edm::pset::OperatorNode*>(node.get());
    
    
      SeqMap::iterator seqIt = sequences.find(onode->left()->name()); 
      if (seqIt!= sequences.end()) {
        onode->left()= seqIt->second->wrapped();
        onode->left()->setParent(onode);
      }
      seqIt = sequences.find(onode->right()->name()); 
      if (seqIt!= sequences.end()){
        onode->right()= seqIt->second->wrapped(); 
        onode->right()->setParent(onode);
      }
      sequenceSubstitution(onode->left(), sequences);
      sequenceSubstitution(onode->right(),sequences);
    
    }// else (operand)
  
  } // sequenceSubstitution


  void ProcessDesc::dumpTree(NodePtr& node){
    if(node->type()=="operand"){ 
      cout << " Operand " << node->name()<< " p:";
      if (node->getParent()) cout <<  node->getParent()->name();cout<< endl;
    } else{	
      edm::pset::OperatorNode* op = dynamic_cast<edm::pset::OperatorNode*>(node.get());
      cout << " Operator: " << op->name()<<"["<<op->type()<<"]" 
	   << " l:" << op->left() << " r:"<<op->right()<< " p:";
      if (op->getParent())cout<<  op->getParent()->name() << endl;
      dumpTree(op->left());
      dumpTree(op->right());
    }
  } // dumpTree


  std::string  ProcessDesc::getDependencies(const std::string& modulename){
    return validator_->dependencies(modulename);
  }


  boost::shared_ptr<edm::ParameterSet>  
  ProcessDesc::getProcessPSet() const{
    return pset_;

  }

  boost::shared_ptr<std::vector<ParameterSet> > 
  ProcessDesc::getServicesPSets() const{
    return services_;
  }
} // namespace edm
