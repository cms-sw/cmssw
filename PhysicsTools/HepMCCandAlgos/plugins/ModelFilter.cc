#include "PhysicsTools/HepMCCandAlgos/interface/ModelFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <cstdlib>

using namespace std;
using namespace edm;

ModelFilter::ModelFilter(const edm::ParameterSet& iConfig) {
  tokenSource_ = consumes<LHEEventProduct>(iConfig.getParameter<InputTag>("source"));
  modelTag_ = iConfig.getParameter<string>("modelTag");
  parameterMins_ = iConfig.getParameter<vector<double> >("parameterMins");
  parameterMaxs_ = iConfig.getParameter<vector<double> >("parameterMaxs");
}

ModelFilter::~ModelFilter() {}

bool ModelFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  Handle<LHEEventProduct> product;
  iEvent.getByToken(tokenSource_, product);
  comments_const_iterator comment;

  string tempString;
  vector<string> parameters;

  for (comment = product->comments_begin(); comment != product->comments_end(); comment++) {
    if (comment->find(modelTag_) != string::npos) {
      tempString = comment->substr(comment->find(modelTag_), comment->size());
      tempString = tempString.substr(0, tempString.find(' '));
      parameters = split(tempString, "_");

      if (parameters.size() - 1 != parameterMins_.size()) {
        std::cout << "Error: number of modeParameters does not match number of parameters in file" << std::endl;
        return false;
      } else if (parameterMins_.size() != parameterMaxs_.size()) {
        std::cout << "Error: umber of parameter mins != number parameter maxes" << std::endl;
      } else {
        for (unsigned i = 0; i < parameterMins_.size(); i++) {
          if (parameterMins_[i] > atof(parameters[i + 1].c_str()) ||
              parameterMaxs_[i] < atof(parameters[i + 1].c_str())) {
            return false;
          }
        }
        return true;
      }
    }
  }
  std::cout << "FAILED: " << *comment << std::endl;
  return false;
}
void ModelFilter::beginJob() {}

void ModelFilter::endJob() {}

void ModelFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
vector<string> ModelFilter::split(string fstring, string splitter) {
  vector<string> returnVector;
  size_t cursor;
  string beforeSplitter;
  string afterSplitter = fstring;
  if (fstring.find(splitter) == string::npos) {
    std::cout << "No " << splitter << " found" << std::endl;
    returnVector.push_back(fstring);
    return returnVector;
  } else {
    while (afterSplitter.find(splitter) != string::npos) {
      cursor = afterSplitter.find(splitter);

      beforeSplitter = afterSplitter.substr(0, cursor);
      afterSplitter = afterSplitter.substr(cursor + 1, afterSplitter.size());

      returnVector.push_back(beforeSplitter);

      if (afterSplitter.find(splitter) == string::npos)
        returnVector.push_back(afterSplitter);
    }
    return returnVector;
  }
}
DEFINE_FWK_MODULE(ModelFilter);
