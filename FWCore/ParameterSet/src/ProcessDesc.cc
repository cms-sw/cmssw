/**
   \file
   Implementation of calss ProcessDesc

   \author Stefano ARGIRO
   \version $Id: ProcessDesc.cc,v 1.8 2006/08/28 19:15:19 rpw Exp $
   \date 17 Jun 2005
*/

static const char CVSId[] = "$Id: ProcessDesc.cc,v 1.8 2006/08/28 19:15:19 rpw Exp $";


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
  : validator_(0),
    pathFragments_(),
    pset_(new ParameterSet),
    services_(new std::vector<ParameterSet>()),
    bookkeeping_()
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
        //FIXME order-dependent
	sequenceSubstitution((*pathIt)->wrapped(), sequences);
	fillPath((*pathIt),triggerpaths);
      }


      if ((*pathIt)->type()=="endpath") {
	sequenceSubstitution((*pathIt)->wrapped(), sequences);
	fillPath((*pathIt),endpaths);
      }
     
     
    } // loop on path fragments

    Strs schedule(findSchedule(triggerpaths, endpaths));

    if(1 <= edm::debugit())
      {
	std::cerr << "\nschedule=\n  ";
	std::copy(schedule.begin(),schedule.end(),
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
    pset_->addParameter("@paths",schedule);
   
    validator_= 
      new ScheduleValidator(pathFragments_,*pset_); 
    validator_->validate();
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
  ProcessDesc::getNames(const edm::pset::Node* n, Strs& out) const {
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

  ProcessDesc::Strs ProcessDesc::findSchedule(ProcessDesc::Strs & triggerPaths,
                                              ProcessDesc::Strs & endPaths) const
  {
    Strs result;
    bool found = false;
    ProcessDesc::PathContainer::const_iterator pathIt;

    for(pathIt= pathFragments_.begin();
        pathIt!=pathFragments_.end(); ++pathIt)
    {

      if ((*pathIt)->type()=="schedule") 
      {
        // no duplicates
        if(found)
        {
          std::ostringstream trace;
          (*pathIt)->printTrace(trace);
          throw edm::Exception(errors::Configuration,"duplicate schedule")
             << "Second schedule statement found at " << trace.str();
        }
        else 
        {
          found = true;
          getNames((*pathIt)->wrapped().get(), result);
          // now override triggerPaths with everything that
          // was in the schedule before the first endpath
            //endOfTriggerPaths = std::find(result.begin(), result.end(), *(endPaths.begin()) );
          Strs::iterator endOfTriggerPaths = std::find_first_of(result.begin(), result.end(),
                                                                endPaths.begin(), endPaths.end());
          // override trigger_paths and endpaths
          triggerPaths = Strs(result.begin(), endOfTriggerPaths);
          endPaths = Strs(endOfTriggerPaths, result.end());
        }
      }
    }

    if(!found)
    {
        // only take defaults if there's only one path and at most one endpath
//        if(triggerPaths.size() > 1 || endPaths.size() > triggerPaths.size())
//        {
//          throw edm::Exception(errors::Configuration,"No schedule")
//             << "More than one path found, so a schedule statement is needed.";
//        }
///        else 
//        {
          // just take defaults
          result = triggerPaths;
          result.insert(result.end(), endPaths.begin(), endPaths.end());
//        }
    }
    return result;
  }




} // namespace edm
