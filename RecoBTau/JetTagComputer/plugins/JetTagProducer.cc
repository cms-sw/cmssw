// -*- C++ -*-
//
// Package:    JetTagProducer
// Class:      JetTagProducer
//
/**\class JetTagProducer JetTagProducer.cc RecoBTag/JetTagProducer/src/JetTagProducer.cc

 Description: Uses a JetTagComputer to produce JetTags from TagInfos

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Thu Apr  6 09:56:23 CEST 2006
//
//

// system include files
#include <memory>
#include <string>
#include <vector>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include <string>
#include <vector>
#include <map>

using namespace std;
using namespace reco;
using namespace edm;

class JetTagProducer : public edm::global::EDProducer<> {
public:
  explicit JetTagProducer(const edm::ParameterSet &);
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  std::vector<edm::EDGetTokenT<edm::View<reco::BaseTagInfo> > > token_tagInfos_;
  const edm::ESGetToken<JetTagComputer, JetTagComputerRecord> token_computer_;
  const edm::EDPutTokenT<JetTagCollection> token_put_;
  mutable std::atomic<unsigned long long> recordCacheID_{0};
  unsigned int nTagInfos_;
};

//
// constructors and destructor
//
JetTagProducer::JetTagProducer(const ParameterSet &iConfig)
    : token_computer_(esConsumes(edm::ESInputTag("", iConfig.getParameter<string>("jetTagComputer")))),
      token_put_(produces()) {
  const std::vector<edm::InputTag> tagInfos = iConfig.getParameter<vector<InputTag> >("tagInfos");
  nTagInfos_ = tagInfos.size();
  token_tagInfos_.reserve(nTagInfos_);
  std::transform(tagInfos.begin(), tagInfos.end(), std::back_inserter(token_tagInfos_), [this](auto const &tagInfo) {
    return consumes<View<BaseTagInfo> >(tagInfo);
  });
}

//
// member functions
//

// map helper - for some reason RefToBase lacks operator < (...)
namespace {
  struct JetRefCompare {
    inline bool operator()(const RefToBase<Jet> &j1, const RefToBase<Jet> &j2) const {
      return j1.id() < j2.id() || (j1.id() == j2.id() && j1.key() < j2.key());
    }
  };
}  // namespace

// ------------ method called to produce the data  ------------
void JetTagProducer::produce(StreamID, Event &iEvent, const EventSetup &iSetup) const {
  auto const &computer = iSetup.getData(token_computer_);

  auto oldID = recordCacheID_.load();
  auto newID = iSetup.get<JetTagComputerRecord>().cacheIdentifier();
  if (oldID != newID) {
    unsigned int nLabels = computer.getInputLabels().size();
    if (nLabels == 0)
      ++nLabels;
    if (nTagInfos_ != nLabels) {
      vector<string> inputLabels(computer.getInputLabels());
      // backward compatible case, use default tagInfo
      if (inputLabels.empty())
        inputLabels.push_back("tagInfo");
      std::string message(
          "VInputTag size mismatch - the following taginfo "
          "labels are needed:\n");
      for (vector<string>::const_iterator iter = inputLabels.begin(); iter != inputLabels.end(); ++iter)
        message += "\"" + *iter + "\"\n";
      throw edm::Exception(errors::Configuration) << message;
    }
    //only if we didn't encounter a problem should we update the cache
    // that way other threads will see the same problem
    recordCacheID_.compare_exchange_strong(oldID, newID);
  }

  // now comes the tricky part:
  // we need to collect all requested TagInfos belonging to the same jet

  typedef vector<const BaseTagInfo *> TagInfoPtrs;
  typedef RefToBase<Jet> JetRef;
  typedef map<JetRef, TagInfoPtrs, JetRefCompare> JetToTagInfoMap;

  JetToTagInfoMap jetToTagInfos;

  // retrieve all requested TagInfos
  Handle<View<BaseTagInfo> > tagInfoHandle;
  for (unsigned int i = 0; i < nTagInfos_; i++) {
    auto tagHandle = iEvent.getHandle(token_tagInfos_[i]);
    // take first tagInfo
    if (not tagInfoHandle.isValid()) {
      tagInfoHandle = tagHandle;
    }

    for (auto const &tagInfo : *tagHandle) {
      TagInfoPtrs &tagInfos = jetToTagInfos[tagInfo.jet()];
      if (tagInfos.empty())
        tagInfos.resize(nTagInfos_);

      tagInfos[i] = &tagInfo;
    }
  }

  std::unique_ptr<JetTagCollection> jetTagCollection;
  if (!tagInfoHandle.product()->empty()) {
    RefToBase<Jet> jj = tagInfoHandle->begin()->jet();
    jetTagCollection = std::make_unique<JetTagCollection>(edm::makeRefToBaseProdFrom(jj, iEvent));
  } else
    jetTagCollection = std::make_unique<JetTagCollection>();

  // now loop over the map and compute all JetTags
  for (JetToTagInfoMap::const_iterator iter = jetToTagInfos.begin(); iter != jetToTagInfos.end(); iter++) {
    const TagInfoPtrs &tagInfoPtrs = iter->second;

    JetTagComputer::TagInfoHelper helper(tagInfoPtrs);
    float discriminator = computer(helper);

    (*jetTagCollection)[iter->first] = discriminator;
  }

  iEvent.put(token_put_, std::move(jetTagCollection));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void JetTagProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("jetTagComputer", "combinedMVAComputer");
  {
    std::vector<edm::InputTag> tagInfos;
    tagInfos.push_back(edm::InputTag("impactParameterTagInfos"));
    tagInfos.push_back(edm::InputTag("inclusiveSecondaryVertexFinderTagInfos"));
    tagInfos.push_back(edm::InputTag("softPFMuonsTagInfos"));
    tagInfos.push_back(edm::InputTag("softPFElectronsTagInfos"));
    desc.add<std::vector<edm::InputTag> >("tagInfos", tagInfos);
  }
  descriptions.addDefault(desc);
}

// define it as plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetTagProducer);
