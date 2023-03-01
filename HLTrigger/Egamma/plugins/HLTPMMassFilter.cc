/** \class HLTPMMassFilter
 *
 * Original Author: Jeremy Werner
 * Institution: Princeton University, USA
 * Contact: Jeremy.Werner@cern.ch
 * Date: February 21, 2007
 */
#include "HLTPMMassFilter.h"

#include <cstdlib>
#include <cmath>

//
// constructors and destructor
//
HLTPMMassFilter::HLTPMMassFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig), magFieldToken_(esConsumes()) {
  candTag_ = iConfig.getParameter<edm::InputTag>("candTag");
  beamSpot_ = iConfig.getParameter<edm::InputTag>("beamSpot");
  l1EGTag_ = iConfig.getParameter<edm::InputTag>("l1EGCand");

  lowerMassCut_ = iConfig.getParameter<double>("lowerMassCut");
  upperMassCut_ = iConfig.getParameter<double>("upperMassCut");
  nZcandcut_ = iConfig.getParameter<int>("nZcandcut");
  reqOppCharge_ = iConfig.getUntrackedParameter<bool>("reqOppCharge", false);
  isElectron1_ = iConfig.getUntrackedParameter<bool>("isElectron1", true);
  isElectron2_ = iConfig.getUntrackedParameter<bool>("isElectron2", true);

  candToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(candTag_);
  beamSpotToken_ = consumes<reco::BeamSpot>(beamSpot_);
}

HLTPMMassFilter::~HLTPMMassFilter() = default;

void HLTPMMassFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag", edm::InputTag("hltL1NonIsoDoublePhotonEt5UpsHcalIsolFilter"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOfflineBeamSpot"));
  desc.add<double>("lowerMassCut", 8.0);
  desc.add<double>("upperMassCut", 11.0);
  desc.add<int>("nZcandcut", 1);
  desc.addUntracked<bool>("reqOppCharge", true);
  desc.addUntracked<bool>("isElectron1", false);
  desc.addUntracked<bool>("isElectron2", false);
  desc.add<edm::InputTag>("l1EGCand", edm::InputTag("hltL1IsoRecoEcalCandidate"));
  descriptions.add("hltPMMassFilter", desc);
}

// ------------ method called to produce the data  ------------
bool HLTPMMassFilter::hltFilter(edm::Event& iEvent,
                                const edm::EventSetup& iSetup,
                                trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  // The filter object
  if (saveTags()) {
    filterproduct.addCollectionTag(l1EGTag_);
  }

  auto const& theMagField = iSetup.getData(magFieldToken_);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken(candToken_, PrevFilterOutput);

  // beam spot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotToken_, recoBeamSpotHandle);
  // gets its position
  const GlobalPoint vertexPos(
      recoBeamSpotHandle->position().x(), recoBeamSpotHandle->position().y(), recoBeamSpotHandle->position().z());

  int n = 0;

  // REMOVED USAGE OF STATIC ARRAYS
  // double px[66];
  // double py[66];
  // double pz[66];
  // double energy[66];
  std::vector<TLorentzVector> pEleCh1;
  std::vector<TLorentzVector> pEleCh2;
  std::vector<double> charge;

  if (isElectron1_ && isElectron2_) {
    Ref<ElectronCollection> refele;

    vector<Ref<ElectronCollection> > electrons;
    PrevFilterOutput->getObjects(TriggerElectron, electrons);

    for (auto& electron : electrons) {
      refele = electron;

      TLorentzVector pThisEle(refele->px(), refele->py(), refele->pz(), refele->energy());
      pEleCh1.push_back(pThisEle);
      charge.push_back(refele->charge());
    }

    std::vector<bool> save_cand(electrons.size(), true);
    for (unsigned int jj = 0; jj < electrons.size(); jj++) {
      for (unsigned int ii = jj + 1; ii < electrons.size(); ii++) {
        if (reqOppCharge_ && charge[jj] * charge[ii] > 0)
          continue;
        if (isGoodPair(pEleCh1[jj], pEleCh1[ii])) {
          n++;
          for (auto const idx : {jj, ii}) {
            if (save_cand[idx])
              filterproduct.addObject(TriggerElectron, electrons[idx]);
            save_cand[idx] = false;
          }
        }
      }
    }

  } else {
    Ref<RecoEcalCandidateCollection> refsc;

    vector<Ref<RecoEcalCandidateCollection> > scs;
    PrevFilterOutput->getObjects(TriggerCluster, scs);
    if (scs.empty())
      PrevFilterOutput->getObjects(TriggerPhoton, scs);  //we dont know if its type trigger cluster or trigger photon

    for (auto& i : scs) {
      refsc = i;
      const reco::SuperClusterRef sc = refsc->superCluster();
      TLorentzVector pscPos = approxMomAtVtx(theMagField, vertexPos, sc, 1);
      pEleCh1.push_back(pscPos);

      TLorentzVector pscEle = approxMomAtVtx(theMagField, vertexPos, sc, -1);
      pEleCh2.push_back(pscEle);
    }

    std::vector<bool> save_cand(scs.size(), true);
    for (unsigned int jj = 0; jj < scs.size(); jj++) {
      for (unsigned int ii = jj + 1; ii < scs.size(); ii++) {
        if (isGoodPair(pEleCh1[jj], pEleCh2[ii]) or isGoodPair(pEleCh2[jj], pEleCh1[ii])) {
          n++;
          for (auto const idx : {jj, ii}) {
            if (save_cand[idx])
              filterproduct.addObject(TriggerCluster, scs[idx]);
            save_cand[idx] = false;
          }
        }
      }
    }
  }

  // filter decision
  bool accept(n >= nZcandcut_);

  return accept;
}

bool HLTPMMassFilter::isGoodPair(TLorentzVector const& v1, TLorentzVector const& v2) const {
  if (std::abs(v1.E() - v2.E()) < 0.00001)
    return false;

  auto const mass = (v1 + v2).M();
  return (mass >= lowerMassCut_ and mass <= upperMassCut_);
}

TLorentzVector HLTPMMassFilter::approxMomAtVtx(const MagneticField& magField,
                                               const GlobalPoint& xvert,
                                               const reco::SuperClusterRef sc,
                                               int charge) const {
  GlobalPoint xsc(sc->position().x(), sc->position().y(), sc->position().z());
  float energy = sc->energy();
  auto theFTS = trackingTools::ftsFromVertexToPoint(magField, xsc, xvert, energy, charge);
  float theApproxMomMod = theFTS.momentum().x() * theFTS.momentum().x() +
                          theFTS.momentum().y() * theFTS.momentum().y() + theFTS.momentum().z() * theFTS.momentum().z();
  TLorentzVector theApproxMom(
      theFTS.momentum().x(), theFTS.momentum().y(), theFTS.momentum().z(), sqrt(theApproxMomMod + 2.61121E-7));
  return theApproxMom;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPMMassFilter);
