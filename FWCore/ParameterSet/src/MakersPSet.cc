#include "FWCore/ParameterSet/interface/Makers.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/ParseTree.h"
#include "FWCore/ParameterSet/interface/CompositeNode.h"

namespace edm {
  namespace pset {

    boost::shared_ptr<edm::ParameterSet>
    makePSet(const std::string & s)
    {
      boost::shared_ptr<edm::ParameterSet> result(new ParameterSet);

      NodePtrListPtr nodeList = edm::pset::parse(s.c_str());
      for(NodePtrList::const_iterator listItr = nodeList->begin(), listItrEnd = nodeList->end();
          listItr != listItrEnd; ++listItr)
      { 
        (**listItr).insertInto(*result);
      }

      return result;
    }

    boost::shared_ptr<edm::ParameterSet>
    makeDefaultPSet(const edm::FileInPath & fip)
    {
      std::string config = read_whole_file(fip.fullPath());
      edm::pset::ParseTree tree(config);
      boost::shared_ptr<ParameterSet> pset(new ParameterSet);
      tree.top()->CompositeNode::insertInto(*pset);
      return pset;
    }
  }
}
