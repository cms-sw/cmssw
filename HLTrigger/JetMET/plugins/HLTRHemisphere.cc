#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/Common/interface/Ref.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TVector3.h"
#include "TLorentzVector.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "HLTrigger/JetMET/interface/HLTRHemisphere.h"

#include <vector>

//
// constructors and destructor
//
HLTRHemisphere::HLTRHemisphere(const edm::ParameterSet& iConfig)
    : inputTag_(iConfig.getParameter<edm::InputTag>("inputTag")),
      muonTag_(iConfig.getParameter<edm::InputTag>("muonTag")),
      doMuonCorrection_(iConfig.getParameter<bool>("doMuonCorrection")),
      muonEta_(iConfig.getParameter<double>("maxMuonEta")),
      min_Jet_Pt_(iConfig.getParameter<double>("minJetPt")),
      max_Eta_(iConfig.getParameter<double>("maxEta")),
      max_NJ_(iConfig.getParameter<int>("maxNJ")),
      accNJJets_(iConfig.getParameter<bool>("acceptNJ")) {
  LogDebug("") << "Input/minJetPt/maxEta/maxNJ/acceptNJ : " << inputTag_.encode() << " " << min_Jet_Pt_ << "/"
               << max_Eta_ << "/" << max_NJ_ << "/" << accNJJets_ << ".";

  m_theJetToken = consumes<edm::View<reco::Jet>>(inputTag_);
  m_theMuonToken = consumes<std::vector<reco::RecoChargedCandidate>>(muonTag_);
  //register your products
  produces<std::vector<math::XYZTLorentzVector>>();
}

HLTRHemisphere::~HLTRHemisphere() = default;

void HLTRHemisphere::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag", edm::InputTag("hltMCJetCorJetIcone5HF07"));
  desc.add<edm::InputTag>("muonTag", edm::InputTag(""));
  desc.add<bool>("doMuonCorrection", false);
  desc.add<double>("maxMuonEta", 2.1);
  desc.add<double>("minJetPt", 30.0);
  desc.add<double>("maxEta", 3.0);
  desc.add<int>("maxNJ", 7);
  desc.add<bool>("acceptNJ", true);
  descriptions.add("hltRHemisphere", desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTRHemisphere::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace math;
  using namespace trigger;

  typedef XYZTLorentzVector LorentzVector;

  // get hold of collection of objects
  //   Handle<CaloJetCollection> jets;
  Handle<View<Jet>> jets;
  iEvent.getByToken(m_theJetToken, jets);

  // get hold of the muons, if necessary
  Handle<vector<reco::RecoChargedCandidate>> muons;
  if (doMuonCorrection_)
    iEvent.getByToken(m_theMuonToken, muons);

  // The output Collection
  std::unique_ptr<vector<math::XYZTLorentzVector>> Hemispheres(new vector<math::XYZTLorentzVector>);

  // look at all objects, check cuts and add to filter object
  int n(0);
  vector<math::XYZTLorentzVector> JETS;
  for (auto const& i : *jets) {
    if (std::abs(i.eta()) < max_Eta_ && i.pt() >= min_Jet_Pt_) {
      JETS.push_back(i.p4());
      n++;
    }
  }

  if (n > max_NJ_ && max_NJ_ != -1) {
    iEvent.put(std::move(Hemispheres));
    return accNJJets_;  // too many jets, accept for timing
  }

  if (doMuonCorrection_) {
    const int nMu = 2;
    int muonIndex[nMu] = {-1, -1};
    std::vector<reco::RecoChargedCandidate>::const_iterator muonIt;
    int index = 0;
    int nPassMu = 0;
    for (muonIt = muons->begin(); muonIt != muons->end(); muonIt++, index++) {
      if (std::abs(muonIt->eta()) > muonEta_ || muonIt->pt() < min_Jet_Pt_)
        continue;                            // skip muons out of eta range or too low pT
      if (nPassMu >= 2) {                    // if we have already accepted two muons, accept the event
        iEvent.put(std::move(Hemispheres));  // too many muons, accept for timing
        return true;
      }
      muonIndex[nPassMu++] = index;
    }
    //muons as MET
    this->ComputeHemispheres(Hemispheres, JETS);
    //lead muon as jet
    if (nPassMu > 0) {
      std::vector<math::XYZTLorentzVector> muonJets;
      reco::RecoChargedCandidate leadMu = muons->at(muonIndex[0]);
      muonJets.push_back(leadMu.p4());
      Hemispheres->push_back(leadMu.p4());
      this->ComputeHemispheres(Hemispheres, JETS, &muonJets);  // lead muon as jet
      if (nPassMu > 1) {                                       // two passing muons
        muonJets.pop_back();
        reco::RecoChargedCandidate secondMu = muons->at(muonIndex[1]);
        muonJets.push_back(secondMu.p4());
        Hemispheres->push_back(secondMu.p4());
        this->ComputeHemispheres(Hemispheres, JETS, &muonJets);  // lead muon as v, second muon as jet
        muonJets.push_back(leadMu.p4());
        this->ComputeHemispheres(Hemispheres, JETS, &muonJets);  // both muon as jets
      }
    }
  } else {  // do MuonCorrection==false
    if (n < 2)
      return false;                               // not enough jets and not adding in muons
    this->ComputeHemispheres(Hemispheres, JETS);  // don't do the muon isolation, just run once and done
  }
  //Format:
  // 0 muon: 2 hemispheres (2)
  // 1 muon: 2 hemisheress + leadMuP4 + 2 hemispheres (5)
  // 2 muon: 2 hemispheres + leadMuP4 + 2 hemispheres + 2ndMuP4 + 4 Hemispheres (10)
  iEvent.put(std::move(Hemispheres));
  return true;
}

void HLTRHemisphere::ComputeHemispheres(std::unique_ptr<std::vector<math::XYZTLorentzVector>>& hlist,
                                        const std::vector<math::XYZTLorentzVector>& JETS,
                                        std::vector<math::XYZTLorentzVector>* extraJets) {
  using namespace math;
  using namespace reco;
  XYZTLorentzVector j1R(0.1, 0., 0., 0.1);
  XYZTLorentzVector j2R(0.1, 0., 0., 0.1);
  int nJets = JETS.size();
  if (extraJets)
    nJets += extraJets->size();

  if (nJets < 2) {  // put empty hemispheres if not enough jets
    hlist->push_back(j1R);
    hlist->push_back(j2R);
    return;
  }
  unsigned int N_comb = pow(2, nJets);  // compute the number of combinations of jets possible
  //Make the hemispheres
  double M_minR = 9999999999.0;
  unsigned int j_count;
  for (unsigned int i = 0; i < N_comb; i++) {
    XYZTLorentzVector j_temp1, j_temp2;
    unsigned int itemp = i;
    j_count = N_comb / 2;
    unsigned int count = 0;
    while (j_count > 0) {
      if (itemp / j_count == 1) {
        if (count < JETS.size())
          j_temp1 += JETS.at(count);
        else
          j_temp1 += extraJets->at(count - JETS.size());
      } else {
        if (count < JETS.size())
          j_temp2 += JETS.at(count);
        else
          j_temp2 += extraJets->at(count - JETS.size());
      }
      itemp -= j_count * (itemp / j_count);
      j_count /= 2;
      count++;
    }
    double M_temp = j_temp1.M2() + j_temp2.M2();
    if (M_temp < M_minR) {
      M_minR = M_temp;
      j1R = j_temp1;
      j2R = j_temp2;
    }
  }

  hlist->push_back(j1R);
  hlist->push_back(j2R);
  return;
}

DEFINE_FWK_MODULE(HLTRHemisphere);
