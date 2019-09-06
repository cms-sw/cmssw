#include <string>
#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMadgraph.h"
#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingAlpgen.h"

namespace gen {

  JetMatching::JetMatching(const edm::ParameterSet& params) { fMatchingStatus = false; }

  JetMatching::~JetMatching() {}

  void JetMatching::init(const lhef::LHERunInfo* runInfo) {}

  void JetMatching::beforeHadronisation(const lhef::LHEEvent* event) {}

  void JetMatching::beforeHadronisationExec() {}

  std::set<std::string> JetMatching::capabilities() const {
    std::set<std::string> result;
    result.insert("psFinalState");
    result.insert("hepmc");
    return result;
  }

  std::unique_ptr<JetMatching> JetMatching::create(const edm::ParameterSet& params) {
    std::string scheme = params.getParameter<std::string>("scheme");

    std::unique_ptr<JetMatching> matching;

    if (scheme == "Madgraph") {
      matching.reset(new JetMatchingMadgraph(params));
    } else if (scheme == "Alpgen") {
      matching.reset(new JetMatchingAlpgen(params));
    } else if (scheme == "MLM") {
      matching.reset();
    } else
      throw cms::Exception("InvalidJetMatching") << "Unknown scheme \"" << scheme
                                                 << "\""
                                                    " specified for parton-shower matching."
                                                 << std::endl;

    if (!matching.get())
      throw cms::Exception("InvalidJetMatching") << "Port of " << scheme << "scheme \""
                                                 << "\""
                                                    " for parton-shower matching is still in progress."
                                                 << std::endl;

    return matching;
  }

}  // namespace gen
