
#include "FWCore/ParameterSet/interface/ParameterDescriptionCases.h"

namespace edm {

  std::unique_ptr<ParameterDescriptionCases<bool> > operator||(
      std::unique_ptr<ParameterDescriptionCases<bool> > left, std::unique_ptr<ParameterDescriptionCases<bool> > right) {
    std::unique_ptr<std::map<bool, edm::value_ptr<ParameterDescriptionNode> > > rightCases = right->caseMap();
    for (std::map<bool, edm::value_ptr<ParameterDescriptionNode> >::const_iterator iter = rightCases->begin(),
                                                                                   iEnd = rightCases->end();
         iter != iEnd;
         ++iter) {
      bool caseValue = iter->first;
      std::unique_ptr<ParameterDescriptionNode> node(iter->second->clone());
      left->insert(caseValue, std::move(node));
    }
    return left;
  }

  std::unique_ptr<ParameterDescriptionCases<int> > operator||(std::unique_ptr<ParameterDescriptionCases<int> > left,
                                                              std::unique_ptr<ParameterDescriptionCases<int> > right) {
    std::unique_ptr<std::map<int, edm::value_ptr<ParameterDescriptionNode> > > rightCases = right->caseMap();
    for (std::map<int, edm::value_ptr<ParameterDescriptionNode> >::const_iterator iter = rightCases->begin(),
                                                                                  iEnd = rightCases->end();
         iter != iEnd;
         ++iter) {
      int caseValue = iter->first;
      std::unique_ptr<ParameterDescriptionNode> node(iter->second->clone());
      left->insert(caseValue, std::move(node));
    }
    return left;
  }

  std::unique_ptr<ParameterDescriptionCases<std::string> > operator||(
      std::unique_ptr<ParameterDescriptionCases<std::string> > left,
      std::unique_ptr<ParameterDescriptionCases<std::string> > right) {
    std::unique_ptr<std::map<std::string, edm::value_ptr<ParameterDescriptionNode> > > rightCases = right->caseMap();
    for (std::map<std::string, edm::value_ptr<ParameterDescriptionNode> >::const_iterator iter = rightCases->begin(),
                                                                                          iEnd = rightCases->end();
         iter != iEnd;
         ++iter) {
      std::string caseValue = iter->first;
      std::unique_ptr<ParameterDescriptionNode> node(iter->second->clone());
      left->insert(caseValue, std::move(node));
    }
    return left;
  }
}  // namespace edm
