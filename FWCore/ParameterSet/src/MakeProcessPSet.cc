#include "FWCore/ParameterSet/interface/Makers.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/Nodes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "boost/shared_ptr.hpp"

#include <string>
#include <vector>
#include <map>
#include <stdexcept>


using namespace std;

namespace edm {

using namespace pset;

namespace {
  
  typedef std::vector<boost::shared_ptr<WrapperNode> > WNodes;
  typedef std::vector<std::string> Strs;
  typedef std::map<std::string, Strs > SeqMap;

  void checkOnePath(const WNodes& n)
  {
    if(n.empty())
      throw edm::Exception(errors::Configuration,"PathError")
	<< "ParemeterSet: problem processing path.\n"
	<< "No Path information given in checkOnePath";
    if(n.size()>1)
      throw edm::Exception(errors::Configuration,"PathError")
	<< "ParemeterSet: problem processing path.\n"
	<< "Only one Path expression allowed at this time";
    if(n[0]->type() != "path")
      throw edm::Exception(errors::Configuration,"PathError")
	<< "ParemeterSet: problem processing path.\n"
	<< "Only Path expressions are allowed at this time";
  }

  void getNames(const Node* n, Strs& out)
  {
    if(n->type()=="operand")
      {
	out.push_back(n->name);
      }
    else if(n->type()=="&")
      {
	throw edm::Exception(errors::LogicError,"PathError")
	  << "Only comma operators in Path expressions are allowed "
	  << "at this time";
      }
    else
      {
	const OperatorNode& op = dynamic_cast<const OperatorNode&>(*n);
	getNames(op.left_.get(),out);
	getNames(op.right_.get(),out);
      }
  }

  void fillPath(const WrapperNode* n, 
		const SeqMap& sequences,
                Strs&   paths,  
		ParameterSet* out)
  {
    Strs unsubst_names;
    getNames(n->wrapped_.get(),unsubst_names);
    
    Strs names;

    // perform sequence substitution
    Strs::iterator nameIt;
    for (nameIt = unsubst_names.begin();  
	 nameIt!= unsubst_names.end(); 
	 ++nameIt){
      SeqMap::const_iterator sequenceIt = sequences.find(*nameIt);
      // if the name found is that of an existing sequence
      // then substitute the sequence for that name
      if (sequenceIt != sequences.end()) {
	names.insert(names.end(),
		     sequenceIt->second.begin(), sequenceIt->second.end());
       
      }  
      else names.push_back(*nameIt);		  
    }//

    out->insert(true,n->name,Entry(names,true));
    paths.push_back(n->name); // add to the list of paths

  } // fillPath(..) 

  void fillSequence(const WrapperNode* n, 
		    SeqMap& sequences){
    
    Strs names;
    getNames(n->wrapped_.get(), names);
    sequences[n->name] = names;

  }

}// namespace

  /**
     Create the ParameterSet that describes the process
     
     \note sequence definitions must come before the path in which 
           they are used (obvious ?) in the configuration file
   */
  boost::shared_ptr<edm::ParameterSet> makeProcessPSet(const std::string& config)
  {
    boost::shared_ptr<edm::pset::NodePtrList> nodelist = 
      edm::pset::parse(config.c_str());
    if(0 == nodelist.get()) {
      throw edm::Exception(errors::Configuration,"MakeProcessError")
	<< "Unable to parse configuration file.\n"
	<< "Please check the error message reported earlier.";
    }
    boost::shared_ptr<ProcessDesc> tmp =
      edm::pset::makeProcess(nodelist);

    //    checkOnePath(tmp->pathFragments_);
    //    fillPath(tmp->pathFragments_[0].get(),&tmp->pset_);
   
    SeqMap sequences;

    // loop on path fragments
    ProcessDesc::PathContainer::iterator pathIt; 
    Strs pathnames;

    for(pathIt= tmp->pathFragments_.begin();
	pathIt!=tmp->pathFragments_.end();
	++pathIt){
      
      if ((*pathIt)->type()=="sequence") 
	fillSequence((*pathIt).get(), sequences);

      if ((*pathIt)->type()=="path") fillPath((*pathIt).get(),
					      sequences,pathnames,&tmp->pset_);

       //cout << "path name = " << (*pathIt)->type() << endl;
      if ((*pathIt)->type()=="endpath")
	{
	   //cout << "processing endpath " << endl;
      	   //(*pathIt)->print(cout);
	   fillPath((*pathIt).get(),
	      sequences,pathnames,&tmp->pset_);
	}
    } // loop on path fragments

    tmp->pset_.insert(true,"@paths",Entry(pathnames,true));

    return boost::shared_ptr<edm::ParameterSet>(new ParameterSet(tmp->pset_));
  } // make ProcessPset

  void write_python_form(edm::pset::Node const& topnode, std::ostream& os)
  {
    // Not yet implemented
  }

}//namespace edm

