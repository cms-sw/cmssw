
/*****************************************************************************
 * Project: CMS detector at the CERN
 *
 * Package: PhysicsTools/TagAndProbe
 *
 *
 * Authors:
 *
 *   Kalanand Mishra, Fermilab - kalanand@fnal.gov
 *
 * Description:
 *   - Matches a given object with other objects using deltaR-matching.
 *   - For example: can match a photon with track within a given deltaR.
 *   - Saves collection of the reference vectors of matched objects.
 * History:
 *
 *
 *****************************************************************************/
////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <memory>
#include <vector>
#include <sstream>

////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////

namespace ovm {
  template <typename T1, typename T2>
  struct StreamCache {
    StreamCache(std::string const& cut, std::string const& match) : objCut_(cut, true), objMatchCut_(match, true) {}
    StringCutObjectSelector<T1, true>
        objCut_;  // lazy parsing, to allow cutting on variables not in reco::Candidate class
    StringCutObjectSelector<T2, true>
        objMatchCut_;  // lazy parsing, to allow cutting on variables not in reco::Candidate class
  };
}  // namespace ovm

template <typename T1, typename T2>
class ObjectViewMatcher : public edm::global::EDProducer<edm::StreamCache<ovm::StreamCache<T1, T2>>> {
public:
  // construction/destruction
  ObjectViewMatcher(const edm::ParameterSet& iConfig);
  ~ObjectViewMatcher() override;

  // member functions
  std::unique_ptr<ovm::StreamCache<T1, T2>> beginStream(edm::StreamID) const override;
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  void endJob() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  // member data
  edm::EDGetTokenT<edm::View<T1>> srcCandsToken_;
  std::vector<edm::EDGetTokenT<edm::View<T2>>> srcObjectsTokens_;
  double deltaRMax_;

  std::string moduleLabel_;
  std::string cut_;
  std::string match_;

  mutable std::atomic<unsigned int> nObjectsTot_;
  mutable std::atomic<unsigned int> nObjectsMatch_;
};

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template <typename T1, typename T2>
ObjectViewMatcher<T1, T2>::ObjectViewMatcher(const edm::ParameterSet& iConfig)
    : srcCandsToken_(this->consumes(iConfig.getParameter<edm::InputTag>("srcObject"))),
      srcObjectsTokens_(edm::vector_transform(
          iConfig.getParameter<std::vector<edm::InputTag>>("srcObjectsToMatch"),
          [this](edm::InputTag const& tag) { return this->template consumes<edm::View<T2>>(tag); })),
      deltaRMax_(iConfig.getParameter<double>("deltaRMax")),
      moduleLabel_(iConfig.getParameter<std::string>("@module_label")),
      cut_(iConfig.getParameter<std::string>("srcObjectSelection")),
      match_(iConfig.getParameter<std::string>("srcObjectsToMatchSelection")),
      nObjectsTot_(0),
      nObjectsMatch_(0) {
  this->template produces<std::vector<T1>>();
}

//______________________________________________________________________________
template <typename T1, typename T2>
ObjectViewMatcher<T1, T2>::~ObjectViewMatcher() {}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template <typename T1, typename T2>
std::unique_ptr<ovm::StreamCache<T1, T2>> ObjectViewMatcher<T1, T2>::beginStream(edm::StreamID) const {
  return std::make_unique<ovm::StreamCache<T1, T2>>(cut_, match_);
}

//______________________________________________________________________________
template <typename T1, typename T2>
void ObjectViewMatcher<T1, T2>::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto cleanObjects = std::make_unique<std::vector<T1>>();

  edm::Handle<edm::View<T1>> candidates;
  iEvent.getByToken(srcCandsToken_, candidates);

  bool* isMatch = new bool[candidates->size()];
  for (unsigned int iObject = 0; iObject < candidates->size(); iObject++)
    isMatch[iObject] = false;

  auto& objCut = this->streamCache(iID)->objCut_;
  auto& objMatchCut = this->streamCache(iID)->objMatchCut_;
  for (unsigned int iSrc = 0; iSrc < srcObjectsTokens_.size(); iSrc++) {
    edm::Handle<edm::View<T2>> objects;
    iEvent.getByToken(srcObjectsTokens_[iSrc], objects);

    if (objects->empty())
      continue;

    for (unsigned int iObject = 0; iObject < candidates->size(); iObject++) {
      const T1& candidate = candidates->at(iObject);
      if (!objCut(candidate))
        continue;

      for (unsigned int iObj = 0; iObj < objects->size(); iObj++) {
        const T2& obj = objects->at(iObj);
        if (!objMatchCut(obj))
          continue;
        double deltaR = reco::deltaR(candidate, obj);
        if (deltaR < deltaRMax_)
          isMatch[iObject] = true;
      }
    }
  }

  unsigned int counter = 0;
  typename edm::View<T1>::const_iterator tIt, endcands = candidates->end();
  for (tIt = candidates->begin(); tIt != endcands; ++tIt, ++counter) {
    if (isMatch[counter])
      cleanObjects->push_back(*tIt);
  }

  nObjectsTot_ += candidates->size();
  nObjectsMatch_ += cleanObjects->size();

  delete[] isMatch;
  iEvent.put(std::move(cleanObjects));
}

//______________________________________________________________________________
template <typename T1, typename T2>
void ObjectViewMatcher<T1, T2>::endJob() {
  std::stringstream ss;
  ss << "nObjectsTot=" << nObjectsTot_ << " nObjectsMatched=" << nObjectsMatch_
     << " fObjectsMatch=" << 100. * (nObjectsMatch_ / (double)nObjectsTot_) << "%\n";
  std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++"
            << "\n"
            << moduleLabel_ << "(ObjectViewMatcher) SUMMARY:\n"
            << ss.str() << "++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
}

//______________________________________________________________________________
template <typename T1, typename T2>
void ObjectViewMatcher<T1, T2>::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription pset;
  pset.add<edm::InputTag>("srcObject");
  pset.add<std::vector<edm::InputTag>>("srcObjectsToMatch");
  pset.add<double>("deltaRMax");
  pset.add<std::string>("srcObjectSelection", "");
  pset.add<std::string>("srcObjectsToMatchSelection", "");

  desc.addDefault(pset);
}

////////////////////////////////////////////////////////////////////////////////
// plugin definition
////////////////////////////////////////////////////////////////////////////////

typedef ObjectViewMatcher<reco::Photon, reco::Track> TrackMatchedPhotonProducer;
typedef ObjectViewMatcher<reco::Jet, reco::Track> TrackMatchedJetProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackMatchedPhotonProducer);
DEFINE_FWK_MODULE(TrackMatchedJetProducer);
