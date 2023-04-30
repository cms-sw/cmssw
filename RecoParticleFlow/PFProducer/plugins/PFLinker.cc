/** \class PFLinker
 *  Producer meant for the Post PF reconstruction.
 *
 *  Fills the GsfElectron, Photon and Muon Ref into the PFCandidate
 *  Produces the ValueMap between GsfElectronRef/Photon/Mupns with PFCandidateRef
 *
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>, F. Beaudette - CERN <Florian.Beaudette@cern.ch>
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/MuonReco/interface/MuonToMuonMap.h"

namespace edm {
  class EventSetup;
}  // namespace edm

class PFLinker : public edm::stream::EDProducer<> {
public:
  explicit PFLinker(const edm::ParameterSet&);

  ~PFLinker() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  template <typename TYPE>
  edm::ValueMap<reco::PFCandidatePtr> fillValueMap(
      edm::Event& event,
      std::string label,
      edm::Handle<TYPE>& inputObjCollection,
      const std::map<edm::Ref<TYPE>, reco::PFCandidatePtr>& mapToTheCandidate,
      const edm::OrphanHandle<reco::PFCandidateCollection>& newPFCandColl) const;

private:
  /// Input PFCandidates
  std::vector<edm::EDGetTokenT<reco::PFCandidateCollection>> inputTagPFCandidates_;

  /// Input GsfElectrons
  edm::EDGetTokenT<reco::GsfElectronCollection> inputTagGsfElectrons_;

  /// Input Photons
  edm::EDGetTokenT<reco::PhotonCollection> inputTagPhotons_;

  /// Input Muons
  edm::InputTag muonTag_;
  edm::EDGetTokenT<reco::MuonCollection> inputTagMuons_;
  edm::EDGetTokenT<reco::MuonToMuonMap> inputTagMuonMap_;
  /// name of output collection of PFCandidate
  std::string nameOutputPF_;

  /// name of output ValueMap electrons
  std::string nameOutputElectronsPF_;

  /// name of output ValueMap photons
  std::string nameOutputPhotonsPF_;

  /// name of output merged ValueMap
  std::string nameOutputMergedPF_;

  /// Flags - if true: References will be towards new collection ; if false to the original one
  bool producePFCandidates_;

  /// Set muon refs and produce the value map?
  bool fillMuonRefs_;

  /// Put Electrons within HGCAL coming from SimPFProducer
  bool forceElectronsInHGCAL_;
};

DEFINE_FWK_MODULE(PFLinker);

PFLinker::PFLinker(const edm::ParameterSet& iConfig) {
  // vector of InputTag; more than 1 is not for RECO, it is for analysis

  std::vector<edm::InputTag> tags = iConfig.getParameter<std::vector<edm::InputTag>>("PFCandidate");
  for (unsigned int i = 0; i < tags.size(); ++i)
    inputTagPFCandidates_.push_back(consumes<reco::PFCandidateCollection>(tags[i]));

  inputTagGsfElectrons_ = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("GsfElectrons"));

  inputTagPhotons_ = consumes<reco::PhotonCollection>(iConfig.getParameter<edm::InputTag>("Photons"));

  muonTag_ = iConfig.getParameter<edm::InputTag>("Muons");
  inputTagMuons_ = consumes<reco::MuonCollection>(edm::InputTag(muonTag_.label()));
  inputTagMuonMap_ = consumes<reco::MuonToMuonMap>(muonTag_);

  nameOutputPF_ = iConfig.getParameter<std::string>("OutputPF");

  nameOutputElectronsPF_ = iConfig.getParameter<std::string>("ValueMapElectrons");

  nameOutputPhotonsPF_ = iConfig.getParameter<std::string>("ValueMapPhotons");

  producePFCandidates_ = iConfig.getParameter<bool>("ProducePFCandidates");

  nameOutputMergedPF_ = iConfig.getParameter<std::string>("ValueMapMerged");

  fillMuonRefs_ = iConfig.getParameter<bool>("FillMuonRefs");

  forceElectronsInHGCAL_ = iConfig.getParameter<bool>("forceElectronsInHGCAL");

  // should not produce PFCandidates and read seve
  if (producePFCandidates_ && inputTagPFCandidates_.size() > 1) {
    edm::LogError("PFLinker")
        << " cannot read several collections of PFCandidates and produce a new collection at the same time. "
        << std::endl;
    assert(false);
  }
  if (producePFCandidates_) {
    produces<reco::PFCandidateCollection>(nameOutputPF_);
  }
  produces<edm::ValueMap<reco::PFCandidatePtr>>(nameOutputElectronsPF_);
  produces<edm::ValueMap<reco::PFCandidatePtr>>(nameOutputPhotonsPF_);
  produces<edm::ValueMap<reco::PFCandidatePtr>>(nameOutputMergedPF_);
  if (fillMuonRefs_)
    produces<edm::ValueMap<reco::PFCandidatePtr>>(muonTag_.label());
}

PFLinker::~PFLinker() { ; }

void PFLinker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pfCandidates_p = std::make_unique<reco::PFCandidateCollection>();

  auto gsfElectrons = iEvent.getHandle(inputTagGsfElectrons_);

  std::map<reco::GsfElectronRef, reco::PFCandidatePtr> electronCandidateMap;

  auto photons = iEvent.getHandle(inputTagPhotons_);
  std::map<reco::PhotonRef, reco::PFCandidatePtr> photonCandidateMap;

  edm::Handle<reco::MuonToMuonMap> muonMap;
  if (fillMuonRefs_)
    muonMap = iEvent.getHandle(inputTagMuonMap_);
  std::map<reco::MuonRef, reco::PFCandidatePtr> muonCandidateMap;

  unsigned nColPF = inputTagPFCandidates_.size();

  for (unsigned icol = 0; icol < nColPF; ++icol) {
    auto pfCandidates = iEvent.getHandle(inputTagPFCandidates_[icol]);
    unsigned ncand = pfCandidates->size();

    for (unsigned i = 0; i < ncand; ++i) {
      edm::Ptr<reco::PFCandidate> candPtr(pfCandidates, i);
      reco::PFCandidate cand(candPtr);

      if (!(cand.energy() > 0))
        continue;

      bool isphoton = cand.particleId() == reco::PFCandidate::gamma && cand.mva_nothing_gamma() > 0.;
      bool iselectron = cand.particleId() == reco::PFCandidate::e;
      // PFCandidates may have a valid MuonRef though they are not muons.
      bool hasNonNullMuonRef = cand.muonRef().isNonnull() && fillMuonRefs_;

      // if not an electron or a photon or a muon just fill the PFCandidate collection
      if (!(isphoton || iselectron || hasNonNullMuonRef)) {
        pfCandidates_p->push_back(cand);
        continue;
      }

      if (hasNonNullMuonRef) {
        reco::MuonRef muRef = (*muonMap)[cand.muonRef()];
        cand.setMuonRef(muRef);
        muonCandidateMap[muRef] = candPtr;
      }

      // if it is an electron. Find the GsfElectron with the same GsfTrack
      if (iselectron) {
        const reco::GsfTrackRef& gsfTrackRef(cand.gsfTrackRef());
        auto itcheck = find_if(gsfElectrons->begin(), gsfElectrons->end(), [&gsfTrackRef](const auto& ele) {
          return (ele.gsfTrack() == gsfTrackRef);
        });
        if (itcheck == gsfElectrons->end()) {
          if (!forceElectronsInHGCAL_) {
            std::ostringstream err;
            err << " Problem in PFLinker: no GsfElectron " << std::endl;
            edm::LogError("PFLinker") << err.str();
          } else {
            LogDebug("PFLinker") << "Forcing an electron pfCandidate at: " << cand.eta() << " in HGCAL" << std::endl;
            pfCandidates_p->push_back(cand);
          }
          continue;  // Watch out ! Continue
        }
        reco::GsfElectronRef electronRef(gsfElectrons, itcheck - gsfElectrons->begin());
        cand.setGsfElectronRef(electronRef);
        cand.setSuperClusterRef(electronRef->superCluster());
        // update energy information since now it is done post-particleFlowTmp
        cand.setEcalEnergy(electronRef->superCluster()->rawEnergy(), electronRef->ecalEnergy());
        cand.setDeltaP(electronRef->p4Error(reco::GsfElectron::P4_COMBINATION));
        cand.setP4(electronRef->p4(reco::GsfElectron::P4_COMBINATION));
        electronCandidateMap[electronRef] = candPtr;
      }

      // if it is a photon, find the one with the same PF super-cluster
      if (isphoton) {
        const reco::SuperClusterRef& scRef(cand.superClusterRef());
        auto itcheck = find_if(
            photons->begin(), photons->end(), [&scRef](const auto& photon) { return photon.superCluster() == scRef; });
        if (itcheck == photons->end()) {
          std::ostringstream err;
          err << " Problem in PFLinker: no Photon " << std::endl;
          edm::LogError("PFLinker") << err.str();
          continue;  // Watch out ! Continue
        }
        reco::PhotonRef photonRef(photons, itcheck - photons->begin());
        cand.setPhotonRef(photonRef);
        cand.setSuperClusterRef(photonRef->superCluster());
        // update energy information since now it is done post-particleFlowTmp
        cand.setEcalEnergy(photonRef->superCluster()->rawEnergy(),
                           photonRef->getCorrectedEnergy(reco::Photon::regression2));
        cand.setDeltaP(photonRef->getCorrectedEnergyError(reco::Photon::regression2));
        cand.setP4(photonRef->p4(reco::Photon::regression2));
        photonCandidateMap[photonRef] = candPtr;
      }

      pfCandidates_p->push_back(cand);
    }
    // save the PFCandidates and get a valid handle
  }
  const edm::OrphanHandle<reco::PFCandidateCollection> pfCandidateRefProd =
      (producePFCandidates_) ? iEvent.put(std::move(pfCandidates_p), nameOutputPF_)
                             : edm::OrphanHandle<reco::PFCandidateCollection>();

  // now make the valuemaps

  edm::ValueMap<reco::PFCandidatePtr> pfMapGsfElectrons = fillValueMap<reco::GsfElectronCollection>(
      iEvent, nameOutputElectronsPF_, gsfElectrons, electronCandidateMap, pfCandidateRefProd);

  edm::ValueMap<reco::PFCandidatePtr> pfMapPhotons = fillValueMap<reco::PhotonCollection>(
      iEvent, nameOutputPhotonsPF_, photons, photonCandidateMap, pfCandidateRefProd);

  edm::ValueMap<reco::PFCandidatePtr> pfMapMuons;

  if (fillMuonRefs_) {
    auto muons = iEvent.getHandle(inputTagMuons_);

    pfMapMuons =
        fillValueMap<reco::MuonCollection>(iEvent, muonTag_.label(), muons, muonCandidateMap, pfCandidateRefProd);
  }

  auto pfMapMerged = std::make_unique<edm::ValueMap<reco::PFCandidatePtr>>();
  edm::ValueMap<reco::PFCandidatePtr>::Filler pfMapMergedFiller(*pfMapMerged);

  *pfMapMerged += pfMapGsfElectrons;
  *pfMapMerged += pfMapPhotons;
  if (fillMuonRefs_)
    *pfMapMerged += pfMapMuons;

  iEvent.put(std::move(pfMapMerged), nameOutputMergedPF_);
}

template <typename TYPE>
edm::ValueMap<reco::PFCandidatePtr> PFLinker::fillValueMap(
    edm::Event& event,
    std::string label,
    edm::Handle<TYPE>& inputObjCollection,
    const std::map<edm::Ref<TYPE>, reco::PFCandidatePtr>& mapToTheCandidate,
    const edm::OrphanHandle<reco::PFCandidateCollection>& newPFCandColl) const {
  auto pfMap_p = std::make_unique<edm::ValueMap<reco::PFCandidatePtr>>();
  edm::ValueMap<reco::PFCandidatePtr>::Filler filler(*pfMap_p);

  typedef typename std::map<edm::Ref<TYPE>, reco::PFCandidatePtr>::const_iterator MapTYPE_it;

  unsigned nObj = inputObjCollection->size();
  std::vector<reco::PFCandidatePtr> values(nObj);

  for (unsigned iobj = 0; iobj < nObj; ++iobj) {
    edm::Ref<TYPE> objRef(inputObjCollection, iobj);
    MapTYPE_it itcheck = mapToTheCandidate.find(objRef);

    reco::PFCandidatePtr candPtr;

    if (itcheck != mapToTheCandidate.end())
      candPtr = producePFCandidates_ ? reco::PFCandidatePtr(newPFCandColl, itcheck->second.key()) : itcheck->second;

    values[iobj] = candPtr;
  }

  filler.insert(inputObjCollection, values.begin(), values.end());
  filler.fill();
  edm::ValueMap<reco::PFCandidatePtr> returnValue = *pfMap_p;
  event.put(std::move(pfMap_p), label);
  return returnValue;
}
