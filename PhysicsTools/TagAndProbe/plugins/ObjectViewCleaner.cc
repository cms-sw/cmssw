
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
 *   - Cleans a given object collection of other
 *     cross-object candidates using deltaR-matching.
 *   - For example: can clean a muon collection by
 *      removing all jets in the muon collection.
 *   - Saves collection of the reference vectors of cleaned objects.
 * History:
 *   Generalized the existing CandViewCleaner
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

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <memory>
#include <vector>
#include <sstream>

////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
namespace ovc {
  template <typename T>
  struct StreamCache {
    StreamCache(std::string const& keep, std::string const& remove)
        : objKeepCut_(keep, true), objRemoveCut_(remove, true) {}
    StringCutObjectSelector<T, true>
        objKeepCut_;  // lazy parsing, to allow cutting on variables not in reco::Candidate class
    StringCutObjectSelector<reco::Candidate, true> objRemoveCut_;  // lazy parsing, to allow cutting on variables
  };
}  // namespace ovc
template <typename T>
class ObjectViewCleaner : public edm::global::EDProducer<edm::StreamCache<ovc::StreamCache<T>>> {
public:
  // construction/destruction
  ObjectViewCleaner(const edm::ParameterSet& iConfig);
  ~ObjectViewCleaner() override;

  // member functions
  std::unique_ptr<ovc::StreamCache<T>> beginStream(edm::StreamID) const override;
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  void endJob() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // member data
  const edm::EDGetTokenT<edm::View<T>> srcCandsToken_;
  const std::vector<edm::EDGetTokenT<edm::View<reco::Candidate>>> srcObjectsTokens_;
  const double deltaRMin_;

  const std::string moduleLabel_;
  const std::string keep_;
  const std::string remove_;

  mutable std::atomic<unsigned int> nObjectsTot_;
  mutable std::atomic<unsigned int> nObjectsClean_;
};

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template <typename T>
ObjectViewCleaner<T>::ObjectViewCleaner(const edm::ParameterSet& iConfig)
    : srcCandsToken_(this->consumes(iConfig.getParameter<edm::InputTag>("srcObject"))),
      srcObjectsTokens_(edm::vector_transform(
          iConfig.getParameter<vector<edm::InputTag>>("srcObjectsToRemove"),
          [this](edm::InputTag const& tag) { return this->template consumes<edm::View<reco::Candidate>>(tag); })),
      deltaRMin_(iConfig.getParameter<double>("deltaRMin")),
      moduleLabel_(iConfig.getParameter<string>("@module_label")),
      keep_(iConfig.getParameter<std::string>("srcObjectSelection")),
      remove_(iConfig.getParameter<std::string>("srcObjectsToRemoveSelection")),
      nObjectsTot_(0),
      nObjectsClean_(0) {
  this->template produces<edm::RefToBaseVector<T>>();
}

//______________________________________________________________________________
template <typename T>
ObjectViewCleaner<T>::~ObjectViewCleaner() {}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::unique_ptr<ovc::StreamCache<T>> ObjectViewCleaner<T>::beginStream(edm::StreamID) const {
  return std::make_unique<ovc::StreamCache<T>>(keep_, remove_);
}
//______________________________________________________________________________
template <typename T>
void ObjectViewCleaner<T>::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto cleanObjects = std::make_unique<edm::RefToBaseVector<T>>();

  edm::Handle<edm::View<T>> candidates;
  iEvent.getByToken(srcCandsToken_, candidates);

  bool* isClean = new bool[candidates->size()];
  for (unsigned int iObject = 0; iObject < candidates->size(); iObject++)
    isClean[iObject] = true;

  auto& objKeepCut = this->streamCache(streamID)->objKeepCut_;
  auto& objRemoveCut = this->streamCache(streamID)->objRemoveCut_;
  for (unsigned int iSrc = 0; iSrc < srcObjectsTokens_.size(); iSrc++) {
    edm::Handle<edm::View<reco::Candidate>> objects;
    iEvent.getByToken(srcObjectsTokens_[iSrc], objects);

    for (unsigned int iObject = 0; iObject < candidates->size(); iObject++) {
      const T& candidate = candidates->at(iObject);
      if (!objKeepCut(candidate))
        isClean[iObject] = false;

      for (unsigned int iObj = 0; iObj < objects->size(); iObj++) {
        const reco::Candidate& obj = objects->at(iObj);
        if (!objRemoveCut(obj))
          continue;

        double deltaR = reco::deltaR(candidate, obj);
        if (deltaR < deltaRMin_)
          isClean[iObject] = false;
      }
    }
  }

  for (unsigned int iObject = 0; iObject < candidates->size(); iObject++)
    if (isClean[iObject])
      cleanObjects->push_back(candidates->refAt(iObject));

  nObjectsTot_ += candidates->size();
  nObjectsClean_ += cleanObjects->size();

  delete[] isClean;
  iEvent.put(std::move(cleanObjects));
}

//______________________________________________________________________________
template <typename T>
void ObjectViewCleaner<T>::endJob() {
  stringstream ss;
  ss << "nObjectsTot=" << nObjectsTot_ << " nObjectsClean=" << nObjectsClean_
     << " fObjectsClean=" << 100. * (nObjectsClean_ / (double)nObjectsTot_) << "%\n";
  edm::LogInfo("ObjectViewCleaner") << "++++++++++++++++++++++++++++++++++++++++++++++++++"
                                    << "\n"
                                    << moduleLabel_ << "(ObjectViewCleaner) SUMMARY:\n"
                                    << ss.str() << "++++++++++++++++++++++++++++++++++++++++++++++++++";
}

template <typename T>
void ObjectViewCleaner<T>::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription pset;
  pset.add<edm::InputTag>("srcObject");
  pset.add<std::vector<edm::InputTag>>("srcObjectsToRemove");
  pset.add<double>("deltaRMin");
  pset.add<std::string>("srcObjectSelection", "");
  pset.add<std::string>("srcObjectsToRemoveSelection", "");
  desc.addDefault(pset);
}
////////////////////////////////////////////////////////////////////////////////
// plugin definition
////////////////////////////////////////////////////////////////////////////////

typedef ObjectViewCleaner<reco::Candidate> CandViewCleaner;
typedef ObjectViewCleaner<reco::Jet> JetViewCleaner;
typedef ObjectViewCleaner<reco::Muon> MuonViewCleaner;
typedef ObjectViewCleaner<reco::GsfElectron> GsfElectronViewCleaner;
typedef ObjectViewCleaner<reco::Electron> ElectronViewCleaner;
typedef ObjectViewCleaner<reco::Photon> PhotonViewCleaner;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CandViewCleaner);
DEFINE_FWK_MODULE(JetViewCleaner);
DEFINE_FWK_MODULE(MuonViewCleaner);
DEFINE_FWK_MODULE(GsfElectronViewCleaner);
DEFINE_FWK_MODULE(ElectronViewCleaner);
DEFINE_FWK_MODULE(PhotonViewCleaner);
