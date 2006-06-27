#include "FWCore/ParameterSet/interface/Makers.h"
#include "FWCore/ParameterSet/interface/parse.h"

namespace edm {
  namespace pset {

    boost::shared_ptr<edm::ParameterSet>
    makePSet(const std::string & s)
    {
      boost::shared_ptr<edm::ParameterSet> result(new ParameterSet);

      NodePtrListPtr nodeList = edm::pset::parse(s.c_str());
      for(NodePtrList::const_iterator listItr = nodeList->begin();
          listItr != nodeList->end(); ++listItr)
      { 
        (**listItr).insertInto(*result);
      }

      return result;
    }
  }
}
