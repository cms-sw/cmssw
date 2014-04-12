
#include "FWCore/ParameterSet/interface/ParameterDescriptionCases.h"


namespace edm {

  std::auto_ptr<ParameterDescriptionCases<bool> >
  operator||(std::auto_ptr<ParameterDescriptionCases<bool> > left,
             std::auto_ptr<ParameterDescriptionCases<bool> > right) {

    std::auto_ptr<std::map<bool, edm::value_ptr<ParameterDescriptionNode> > > rightCases = right->caseMap();
    for (std::map<bool, edm::value_ptr<ParameterDescriptionNode> >::const_iterator iter = rightCases->begin(),
	                                                                           iEnd = rightCases->end();
         iter != iEnd; ++iter) {
      bool caseValue = iter->first;
      std::auto_ptr<ParameterDescriptionNode> node(iter->second->clone());
      left->insert(caseValue, node);
    }
    return left;
  }

  std::auto_ptr<ParameterDescriptionCases<int> >
  operator||(std::auto_ptr<ParameterDescriptionCases<int> > left,
             std::auto_ptr<ParameterDescriptionCases<int> > right) {

    std::auto_ptr<std::map<int, edm::value_ptr<ParameterDescriptionNode> > > rightCases = right->caseMap();
    for (std::map<int, edm::value_ptr<ParameterDescriptionNode> >::const_iterator iter = rightCases->begin(),
	                                                                          iEnd = rightCases->end();
         iter != iEnd; ++iter) {
      int caseValue = iter->first;
      std::auto_ptr<ParameterDescriptionNode> node(iter->second->clone());
      left->insert(caseValue, node);
    }
    return left;
  }

  std::auto_ptr<ParameterDescriptionCases<std::string> >
  operator||(std::auto_ptr<ParameterDescriptionCases<std::string> > left,
             std::auto_ptr<ParameterDescriptionCases<std::string> > right) {

    std::auto_ptr<std::map<std::string, edm::value_ptr<ParameterDescriptionNode> > > rightCases = right->caseMap();
    for (std::map<std::string, edm::value_ptr<ParameterDescriptionNode> >::const_iterator iter = rightCases->begin(),
	                                                                        iEnd = rightCases->end();
         iter != iEnd; ++iter) {
      std::string caseValue = iter->first;
      std::auto_ptr<ParameterDescriptionNode> node(iter->second->clone());
      left->insert(caseValue, node);
    }
    return left;
  }
}
