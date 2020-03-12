
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
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
template <typename T1, typename T2>
class ObjectViewMatcher : public edm::EDProducer {
public:
  // construction/destruction
  ObjectViewMatcher(const edm::ParameterSet& iConfig);
  ~ObjectViewMatcher() override;

  // member functions
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void endJob() override;

private:
  // member data
  edm::EDGetTokenT<edm::View<T1>> srcCandsToken_;
  std::vector<edm::EDGetTokenT<edm::View<T2>>> srcObjectsTokens_;
  double deltaRMax_;

  std::string moduleLabel_;

  StringCutObjectSelector<T1, true> objCut_;  // lazy parsing, to allow cutting on variables not in reco::Candidate class
  StringCutObjectSelector<T2, true>
      objMatchCut_;  // lazy parsing, to allow cutting on variables not in reco::Candidate class

  unsigned int nObjectsTot_;
  unsigned int nObjectsMatch_;
};

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template <typename T1, typename T2>
ObjectViewMatcher<T1, T2>::ObjectViewMatcher(const edm::ParameterSet& iConfig)
    : srcCandsToken_(consumes<edm::View<T1>>(iConfig.getParameter<edm::InputTag>("srcObject"))),
      srcObjectsTokens_(
          edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag>>("srcObjectsToMatch"),
                                [this](edm::InputTag const& tag) { return consumes<edm::View<T2>>(tag); })),
      deltaRMax_(iConfig.getParameter<double>("deltaRMax")),
      moduleLabel_(iConfig.getParameter<std::string>("@module_label")),
      objCut_(iConfig.existsAs<std::string>("srcObjectSelection")
                  ? iConfig.getParameter<std::string>("srcObjectSelection")
                  : "",
              true),
      objMatchCut_(iConfig.existsAs<std::string>("srcObjectsToMatchSelection")
                       ? iConfig.getParameter<std::string>("srcObjectsToMatchSelection")
                       : "",
                   true),
      nObjectsTot_(0),
      nObjectsMatch_(0) {
  produces<std::vector<T1>>();
}

//______________________________________________________________________________
template <typename T1, typename T2>
ObjectViewMatcher<T1, T2>::~ObjectViewMatcher() {}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template <typename T1, typename T2>
void ObjectViewMatcher<T1, T2>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto cleanObjects = std::make_unique<std::vector<T1>>();

  edm::Handle<edm::View<T1>> candidates;
  iEvent.getByToken(srcCandsToken_, candidates);

  bool* isMatch = new bool[candidates->size()];
  for (unsigned int iObject = 0; iObject < candidates->size(); iObject++)
    isMatch[iObject] = false;

  for (unsigned int iSrc = 0; iSrc < srcObjectsTokens_.size(); iSrc++) {
    edm::Handle<edm::View<T2>> objects;
    iEvent.getByToken(srcObjectsTokens_[iSrc], objects);

    if (objects->empty())
      continue;

    for (unsigned int iObject = 0; iObject < candidates->size(); iObject++) {
      const T1& candidate = candidates->at(iObject);
      if (!objCut_(candidate))
        continue;

      for (unsigned int iObj = 0; iObj < objects->size(); iObj++) {
        const T2& obj = objects->at(iObj);
        if (!objMatchCut_(obj))
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

////////////////////////////////////////////////////////////////////////////////
// plugin definition
////////////////////////////////////////////////////////////////////////////////

typedef ObjectViewMatcher<reco::Photon, reco::Track> TrackMatchedPhotonProducer;
typedef ObjectViewMatcher<reco::Jet, reco::Track> TrackMatchedJetProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackMatchedPhotonProducer);
DEFINE_FWK_MODULE(TrackMatchedJetProducer);
