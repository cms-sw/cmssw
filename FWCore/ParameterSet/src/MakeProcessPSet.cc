
#include "FWCore/ParameterSet/interface/Makers.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/Nodes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/Entry.h"

#include "boost/shared_ptr.hpp"
#include <string>
#include <vector>
#include <stdexcept>

using namespace std;

namespace edm {

using namespace pset;

namespace {
  
  typedef std::vector<boost::shared_ptr<WrapperNode> > WNodes;
  typedef std::vector<std::string> Strs;

  void checkOnePath(const WNodes& n)
  {
    if(n.empty())
      throw runtime_error("No Path information given");
    if(n.size()>1)
      throw runtime_error("Only one Path expression allowed at this time");
    if(n[0]->type() != "path")
      throw runtime_error("Only Path expressions are allowed at this time");
  }

  void getNames(const Node* n, Strs& out)
  {
    if(n->type()=="operand")
      {
	out.push_back(n->name());
      }
    else if(n->type()=="&")
      {
	throw logic_error("Only comma operators in Path expressions are allowed at this time");
      }
    else
      {
	const OperatorNode& op = dynamic_cast<const OperatorNode&>(*n);
	getNames(op.left_.get(),out);
	getNames(op.right_.get(),out);
      }
  }

  void fillPath(const WrapperNode* n, ParameterSet* out)
  {
    Strs names;
    getNames(n->wrapped_.get(),names);
    out->insert(true,"temporary_single_path",Entry(names,true));
  } 

}

  boost::shared_ptr<edm::ParameterSet> makeProcessPSet(const std::string& config)
  {
    boost::shared_ptr<edm::pset::NodePtrList> nodelist = 
      edm::pset::parse(config.c_str());
    boost::shared_ptr<ProcessDesc> tmp =
      edm::pset::makeProcess(nodelist);

    checkOnePath(tmp->pathFragments_);
    fillPath(tmp->pathFragments_[0].get(),&tmp->pset_);

    return boost::shared_ptr<edm::ParameterSet>(new ParameterSet(tmp->pset_));
  }
}
