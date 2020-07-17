// -*- C++ -*-
//
// Package:    RecoBTag/SecondaryVertex
// Class:      BTagProbabilityToDiscriminator
//
/**
 *
 * Description: EDProducer that performs simple arithmetic on the
 * multi-classifier probabilities to compute simple discriminators
 *
 * Implementation:
 *    A collection of output discriminators is defined in a VPSet, each
 * containing the output name, input probabilities and normalization (empty
 * vInputTag if none) the output is computed as
 *         sum(INPUTS)/sum(normalizations)
 */
//
// Original Author:  Mauro Verzetti (CERN)
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

// from lwtnn
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <unordered_map>
using namespace std;
using namespace reco;
//
// class declaration
//

class BTagProbabilityToDiscriminator : public edm::stream::EDProducer<> {
public:
  explicit BTagProbabilityToDiscriminator(const edm::ParameterSet &);
  ~BTagProbabilityToDiscriminator() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  typedef std::vector<edm::InputTag> vInputTag;
  typedef std::vector<std::string> vstring;
  typedef std::vector<edm::ParameterSet> vPSet;
  struct Discriminator {
    std::string name;  // needed?
    vstring numerator;
    vstring denominator;
  };

  void beginStream(edm::StreamID) override {}
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endStream() override {}

  // ----------member data ---------------------------
  std::vector<Discriminator> discrims_;
  std::unordered_map<std::string, edm::EDGetTokenT<JetTagCollection>> jet_tags_;  // caches jet tags to avoid repetitions
};

BTagProbabilityToDiscriminator::BTagProbabilityToDiscriminator(const edm::ParameterSet &iConfig) {
  for (const auto& discriminator : iConfig.getParameter<vPSet>("discriminators")) {
    Discriminator current;
    current.name = discriminator.getParameter<std::string>("name");
    produces<JetTagCollection>(current.name);

    for (const auto& intag : discriminator.getParameter<vInputTag>("numerator")) {
      if (jet_tags_.find(intag.encode()) == jet_tags_.end()) {  // new
                                                                // probability
        jet_tags_[intag.encode()] = consumes<JetTagCollection>(intag);
      }
      current.numerator.push_back(intag.encode());
    }

    for (const auto& intag : discriminator.getParameter<vInputTag>("denominator")) {
      if (jet_tags_.find(intag.encode()) == jet_tags_.end()) {  // new
                                                                // probability
        jet_tags_[intag.encode()] = consumes<JetTagCollection>(intag);
      }
      current.denominator.push_back(intag.encode());
    }
    discrims_.push_back(current);
  }

  if (jet_tags_.empty()) {
    throw cms::Exception("RuntimeError") << "The module BTagProbabilityToDiscriminator is run without any input "
                                            "probability to work on!"
                                         << std::endl;
  }
}

void BTagProbabilityToDiscriminator::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::unordered_map<std::string, edm::Handle<JetTagCollection>> tags;  // caches jet tags to avoid repetitions
  size_t size = 0;
  bool first = true;
  for (const auto &entry : jet_tags_) {
    edm::Handle<JetTagCollection> tmp;
    iEvent.getByToken(entry.second, tmp);
    tags[entry.first] = tmp;
    if (first)
      size = tmp->size();
    else {
      if (tmp->size() != size) {
        throw cms::Exception("RuntimeError") << "The length of one of the input jet tag collections does not "
                                                "match "
                                             << "with the others, this is probably due to the probabilities "
                                                "belonging to different jet collections, which is forbidden!"
                                             << std::endl;
      }
    }
    first = false;
  }

  // create the output collection
  // which is a "map" RefToBase<Jet> --> float
  vector<std::unique_ptr<JetTagCollection>> output_tags;
  output_tags.reserve(discrims_.size());
  for (size_t i = 0; i < discrims_.size(); ++i) {
    output_tags.push_back(
        std::make_unique<JetTagCollection>(*(tags.begin()->second))  // clone from the first element, will change
                                                                     // the content later on
    );
  }

  // loop over jets
  for (size_t idx = 0; idx < output_tags[0]->size(); idx++) {
    auto key = output_tags[0]->key(idx);  // use key only for writing
    // loop over new discriminators to produce
    for (size_t disc_idx = 0; disc_idx < output_tags.size(); disc_idx++) {
      float numerator = 0;
      for (auto &num : discrims_[disc_idx].numerator)
        numerator += (*tags[num])[idx].second;
      float denominator = !discrims_[disc_idx].denominator.empty() ? 0 : 1;
      for (auto &den : discrims_[disc_idx].denominator)
        denominator += (*tags[den])[idx].second;
      //protect against 0 denominator and undefined jet values (numerator probability < 0)
      float new_value = (denominator != 0 && numerator >= 0) ? numerator / denominator : -10.;
      (*output_tags[disc_idx])[key] = new_value;
    }
  }

  // put the output in the event
  for (size_t i = 0; i < output_tags.size(); ++i) {
    iEvent.put(std::move(output_tags[i]), discrims_[i].name);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the
// module  ------------
void BTagProbabilityToDiscriminator::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription vpsd1;
    vpsd1.add<std::vector<edm::InputTag>>("denominator", {});
    vpsd1.add<std::vector<edm::InputTag>>("numerator",
                                          {
                                              edm::InputTag("pfDeepCSVJetTags", "probb"),
                                              edm::InputTag("pfDeepCSVJetTags", "probbb"),
                                          });
    vpsd1.add<std::string>("name", "BvsAll");
    std::vector<edm::ParameterSet> temp1;
    temp1.reserve(3);
    {
      edm::ParameterSet temp2;
      temp2.addParameter<std::vector<edm::InputTag>>("denominator", {});
      temp2.addParameter<std::vector<edm::InputTag>>("numerator",
                                                     {
                                                         edm::InputTag("pfDeepCSVJetTags", "probb"),
                                                         edm::InputTag("pfDeepCSVJetTags", "probbb"),
                                                     });
      temp2.addParameter<std::string>("name", "BvsAll");
      temp1.push_back(temp2);
    }
    {
      edm::ParameterSet temp2;
      temp2.addParameter<std::vector<edm::InputTag>>("denominator",
                                                     {
                                                         edm::InputTag("pfDeepCSVJetTags", "probc"),
                                                         edm::InputTag("pfDeepCSVJetTags", "probb"),
                                                         edm::InputTag("pfDeepCSVJetTags", "probbb"),
                                                     });
      temp2.addParameter<std::vector<edm::InputTag>>("numerator",
                                                     {
                                                         edm::InputTag("pfDeepCSVJetTags", "probc"),
                                                     });
      temp2.addParameter<std::string>("name", "CvsB");
      temp1.push_back(temp2);
    }
    {
      edm::ParameterSet temp2;
      temp2.addParameter<std::vector<edm::InputTag>>("denominator",
                                                     {
                                                         edm::InputTag("pfDeepCSVJetTags", "probudsg"),
                                                         edm::InputTag("pfDeepCSVJetTags", "probc"),
                                                     });
      temp2.addParameter<std::vector<edm::InputTag>>("numerator",
                                                     {
                                                         edm::InputTag("pfDeepCSVJetTags", "probc"),
                                                     });
      temp2.addParameter<std::string>("name", "CvsL");
      temp1.push_back(temp2);
    }
    desc.addVPSet("discriminators", vpsd1, temp1);
  }
  descriptions.addDefault(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(BTagProbabilityToDiscriminator);
