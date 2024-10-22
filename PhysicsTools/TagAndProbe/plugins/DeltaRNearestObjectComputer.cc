/*****************************************************************************
 * Project: CMS detector at the CERN
 *
 * Package: PhysicsTools/TagAndProbe
 *
 *
 * Authors:
 *   Giovanni Petrucciani, UCSD - Giovanni.Petrucciani@cern.ch
 *
 * Description:
 *   - Matches a given object with other objects using deltaR-matching.
 *   - For example: can match a photon with track within a given deltaR.
 *   - Saves collection of the reference vectors of matched objects.
 * History:
 *
 * Kalanand Mishra, Fermilab - kalanand@fnal.gov
 * Extended the class to compute deltaR with respect to any object
 * (i.e., Candidate, Jet, Muon, Electron, or Photon). The previous
 * version of this class could deltaR only with respect to reco::Candidates.
 * This didn't work if one wanted to apply selection cuts on the Candidate's
 * RefToBase object.
 *****************************************************************************/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

template <typename T>
class DeltaRNearestObjectComputer : public edm::stream::EDProducer<> {
public:
  explicit DeltaRNearestObjectComputer(const edm::ParameterSet& iConfig);
  ~DeltaRNearestObjectComputer() override;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<edm::View<reco::Candidate>> probesToken_;
  edm::EDGetTokenT<edm::View<T>> objectsToken_;
  StringCutObjectSelector<T, true> objCut_;  // lazy parsing, to allow cutting on variables not in reco::Candidate class
};

template <typename T>
DeltaRNearestObjectComputer<T>::DeltaRNearestObjectComputer(const edm::ParameterSet& iConfig)
    : probesToken_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("probes"))),
      objectsToken_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("objects"))),
      objCut_(
          iConfig.existsAs<std::string>("objectSelection") ? iConfig.getParameter<std::string>("objectSelection") : "",
          true) {
  produces<edm::ValueMap<float>>();
}

template <typename T>
DeltaRNearestObjectComputer<T>::~DeltaRNearestObjectComputer() {}

template <typename T>
void DeltaRNearestObjectComputer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // read input
  Handle<View<reco::Candidate>> probes;
  iEvent.getByToken(probesToken_, probes);

  Handle<View<T>> objects;
  iEvent.getByToken(objectsToken_, objects);

  // prepare vector for output
  std::vector<float> values;

  // fill
  View<reco::Candidate>::const_iterator probe, endprobes = probes->end();
  for (probe = probes->begin(); probe != endprobes; ++probe) {
    double dr2min = 10000;
    for (unsigned int iObj = 0; iObj < objects->size(); iObj++) {
      const T& obj = objects->at(iObj);
      if (!objCut_(obj))
        continue;
      double dr2 = deltaR2(*probe, obj);
      if (dr2 < dr2min) {
        dr2min = dr2;
      }
    }
    values.push_back(sqrt(dr2min));
  }

  // convert into ValueMap and store
  auto valMap = std::make_unique<ValueMap<float>>();
  ValueMap<float>::Filler filler(*valMap);
  filler.insert(probes, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap));
}

////////////////////////////////////////////////////////////////////////////////
// plugin definition
////////////////////////////////////////////////////////////////////////////////

typedef DeltaRNearestObjectComputer<reco::Candidate> DeltaRNearestCandidateComputer;
typedef DeltaRNearestObjectComputer<reco::Muon> DeltaRNearestMuonComputer;
typedef DeltaRNearestObjectComputer<reco::Electron> DeltaRNearestElectronComputer;
typedef DeltaRNearestObjectComputer<reco::GsfElectron> DeltaRNearestGsfElectronComputer;
typedef DeltaRNearestObjectComputer<reco::Photon> DeltaRNearestPhotonComputer;
typedef DeltaRNearestObjectComputer<reco::Jet> DeltaRNearestJetComputer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeltaRNearestCandidateComputer);
DEFINE_FWK_MODULE(DeltaRNearestMuonComputer);
DEFINE_FWK_MODULE(DeltaRNearestElectronComputer);
DEFINE_FWK_MODULE(DeltaRNearestGsfElectronComputer);
DEFINE_FWK_MODULE(DeltaRNearestPhotonComputer);
DEFINE_FWK_MODULE(DeltaRNearestJetComputer);
