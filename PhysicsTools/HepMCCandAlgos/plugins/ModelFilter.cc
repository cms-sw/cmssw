#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <cstdlib>

/**
    The ModelFilter class will select events in a "soup" MC
    (like the SUSY signal MC) from the comments of LHEEventProduct
    that match "modelTag". The user can require the value of that
    parameter to lie between a min and max value.
 */

namespace edm {

  class ModelFilter : public edm::global::EDFilter<> {
  public:
    explicit ModelFilter(const edm::ParameterSet&);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    typedef std::vector<std::string>::const_iterator comments_const_iterator;

  private:
    bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
    static std::vector<std::string> split(std::string const& fstring, std::string const& splitter);

    edm::EDGetTokenT<LHEEventProduct> tokenSource_;
    std::string modelTag_;
    std::vector<double> parameterMins_;
    std::vector<double> parameterMaxs_;
  };

}  // namespace edm

using namespace std;
using namespace edm;

ModelFilter::ModelFilter(const edm::ParameterSet& iConfig) {
  tokenSource_ = consumes<LHEEventProduct>(iConfig.getParameter<InputTag>("source"));
  modelTag_ = iConfig.getParameter<string>("modelTag");
  parameterMins_ = iConfig.getParameter<vector<double> >("parameterMins");
  parameterMaxs_ = iConfig.getParameter<vector<double> >("parameterMaxs");
}

bool ModelFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
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
        edm::LogError("ModelFilter") << "number of modeParameters does not match number of parameters in file";
        return false;
      } else if (parameterMins_.size() != parameterMaxs_.size()) {
        edm::LogError("ModelFilter") << "Error: umber of parameter mins != number parameter maxes";
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
  edm::LogInfo("ModelFilter") << "FAILED: " << *comment;
  return false;
}
void ModelFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<InputTag>("source");
  desc.add<string>("modelTag");
  desc.add<vector<double> >("parameterMins");
  desc.add<vector<double> >("parameterMaxs");

  descriptions.addDefault(desc);
}
vector<string> ModelFilter::split(string const& fstring, string const& splitter) {
  vector<string> returnVector;
  size_t cursor;
  string beforeSplitter;
  string afterSplitter = fstring;
  if (fstring.find(splitter) == string::npos) {
    edm::LogInfo("ModelFilter") << "No " << splitter << " found";
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
