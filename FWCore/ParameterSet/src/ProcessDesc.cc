/**
   \file
   Implementation of calss ProcessDesc

   \author Stefano ARGIRO
   \version $Id: ProcessDesc.cc,v 1.16 2007/06/15 18:41:49 wdd Exp $
   \date 17 Jun 2005
*/

static const char CVSId[] = "$Id: ProcessDesc.cc,v 1.16 2007/06/15 18:41:49 wdd Exp $";


#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/ParseTree.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Entry.h"

#include "FWCore/ParameterSet/src/ScheduleValidator.h"
#include "FWCore/ParameterSet/interface/OperatorNode.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include "FWCore/ParameterSet/interface/PSetNode.h"

#include <map>
#include <iostream>
#include <algorithm>
#include <iterator>

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
    edm::pset::ParseTree parsetree(config.c_str());
    parsetree.getProcessNode()->fillProcess(*this);


    writeBookkeeping("@all_modules");
    writeBookkeeping("@all_sources");
    writeBookkeeping("@all_loopers");
    writeBookkeeping("@all_esmodules");
    writeBookkeeping("@all_essources");
    writeBookkeeping("@all_esprefers");

    SeqMap sequences;

    // loop on path fragments
    Strs endpaths, triggerpaths;
   
    for(ProcessDesc::PathContainer::iterator pathIt = pathFragments_.begin(),
					     pathItEnd = pathFragments_.end();
	pathIt != pathItEnd;
	++pathIt) {
     
      if ((*pathIt)->type() == "sequence") {
	sequences[(*pathIt)->name()]= (*pathIt);
      }
     
      if ((*pathIt)->type() == "path") {
        //FIXME order-dependent
	sequenceSubstitution((*pathIt)->wrapped(), sequences);
	fillPath((*pathIt),triggerpaths);
      }


      if ((*pathIt)->type() == "endpath") {
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

    // It is very important that the @trigger_paths parameter set only
    // contain one parameter because the streamer input module needs to
    // be able to recreate it based on the header in the streamer files.
    // The recreated version must have the same ParameterSetID
    ParameterSet paths_trig;
    paths_trig.addParameter("@trigger_paths", triggerpaths);

    pset_->addUntrackedParameter("@trigger_paths",paths_trig);
    pset_->addParameter("@end_paths", endpaths);
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
    if(n->type() == "operand") { 
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
						SeqMap&  sequences) {
  
    if (node->type() == "operand") {
      SeqMap::iterator seqIt = sequences.find(node->name()); 
      if (seqIt != sequences.end()) {
        node = seqIt->second->wrapped();
        sequenceSubstitution(node, sequences);
      }
    } // if operator
    else {
      edm::pset::OperatorNode* onode = dynamic_cast<edm::pset::OperatorNode*>(node.get());
    
      SeqMap::iterator seqIt = sequences.find(onode->left()->name()); 
      if (seqIt != sequences.end()) {
        //onode->left()= seqIt->second->wrapped();
        onode->left()= NodePtr(seqIt->second->wrapped()->clone());
        onode->left()->setParent(onode);
      }
      seqIt = sequences.find(onode->right()->name()); 
      if (seqIt != sequences.end()) {
        //onode->right()= seqIt->second->wrapped(); 
        onode->right()= NodePtr(seqIt->second->wrapped()->clone());
        onode->right()->setParent(onode);
      }
      sequenceSubstitution(onode->left(), sequences);
      sequenceSubstitution(onode->right(),sequences);
    
    }// else (operand)
  
  } // sequenceSubstitution


  void ProcessDesc::dumpTree(NodePtr& node) {
    if(node->type() == "operand") { 
      std::cout << " Operand " << node->name() << " p:";
      if (node->getParent()) std::cout << node->getParent()->name(); std::cout<< std::endl;
    } else {	
      edm::pset::OperatorNode* op = dynamic_cast<edm::pset::OperatorNode*>(node.get());
      std::cout << " Operator: " << op->name() << "[" << op->type() << "]" 
	        << " l:" << op->left() << " r:" << op->right() << " p:";
      if (op->getParent()) std::cout <<  op->getParent()->name() << std::endl;
      dumpTree(op->left());
      dumpTree(op->right());
    }
  } // dumpTree


  std::string  ProcessDesc::getDependencies(const std::string& modulename) {
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

    for(ProcessDesc::PathContainer::const_iterator pathIt = pathFragments_.begin(),
						   pathItEnd = pathFragments_.end();
        pathIt != pathItEnd; ++pathIt)
    {

      if ((*pathIt)->type() == "schedule") 
      {
        // no duplicates
        if(found)
        {
          throw edm::Exception(errors::Configuration,"duplicate schedule")
             << "Second schedule statement found at " << (*pathIt)->traceback();
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
