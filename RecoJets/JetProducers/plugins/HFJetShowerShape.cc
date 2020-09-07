// -*- C++ -*-
//
// Package:    HFJetShowerShape/HFJetShowerShape
// Class:      HFJetShowerShape
//
/**\class HFJetShowerShape HFJetShowerShape.cc HFJetShowerShape/HFJetShowerShape/plugins/HFJetShowerShape.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Laurent Thomas
//         Created:  Tue, 25 Aug 2020 09:17:42 GMT
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

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <iostream>

//
// class declaration
//

class HFJetShowerShape : public edm::stream::EDProducer<> {
public:
  explicit HFJetShowerShape(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  template <typename T>
  void putInEvent(const std::string&, const edm::Handle<edm::View<reco::Jet>>&, std::vector<T>, edm::Event&) const;

  const edm::EDGetTokenT<edm::View<reco::Jet>> jets_token_;
  const edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_token_;

  //Jet pt/eta thresholds
  const double jetPtThreshold_, jetEtaThreshold_;
  //HF geometry
  const double hfTowerEtaWidth_, hfTowerPhiWidth_;
  //Variables for PU subtraction
  const double vertexRecoEffcy_, offsetPerPU_, jetReferenceRadius_;
  //Pt thresholds for showershape variable calculation
  const double stripPtThreshold_, widthPtThreshold_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HFJetShowerShape::HFJetShowerShape(const edm::ParameterSet& iConfig)
    : jets_token_(consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
      vertices_token_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("vertices"))),
      jetPtThreshold_(iConfig.getParameter<double>("jetPtThreshold")),
      jetEtaThreshold_(iConfig.getParameter<double>("jetEtaThreshold")),
      hfTowerEtaWidth_(iConfig.getParameter<double>("hfTowerEtaWidth")),
      hfTowerPhiWidth_(iConfig.getParameter<double>("hfTowerPhiWidth")),
      vertexRecoEffcy_(iConfig.getParameter<double>("vertexRecoEffcy")),
      offsetPerPU_(iConfig.getParameter<double>("offsetPerPU")),
      jetReferenceRadius_(iConfig.getParameter<double>("jetReferenceRadius")),
      stripPtThreshold_(iConfig.getParameter<double>("stripPtThreshold")),
      widthPtThreshold_(iConfig.getParameter<double>("widthPtThreshold")) {
  produces<edm::ValueMap<float>>("sigmaEtaEta");
  produces<edm::ValueMap<float>>("sigmaPhiPhi");
  produces<edm::ValueMap<int>>("centralEtaStripSize");
  produces<edm::ValueMap<int>>("adjacentEtaStripsSize");
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HFJetShowerShape::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //Jets
  auto theJets = iEvent.getHandle(jets_token_);

  //Vertices
  int nPV = iEvent.get(vertices_token_).size();

  //Products
  std::vector<float> v_sigmaEtaEta, v_sigmaPhiPhi;
  v_sigmaEtaEta.reserve(theJets->size());
  v_sigmaPhiPhi.reserve(theJets->size());
  std::vector<int> v_size_CentralEtaStrip, v_size_AdjacentEtaStrips;
  v_size_CentralEtaStrip.reserve(theJets->size());
  v_size_AdjacentEtaStrips.reserve(theJets->size());

  //Et offset for HF PF candidates
  double puoffset = offsetPerPU_ / (M_PI * jetReferenceRadius_ * jetReferenceRadius_) * nPV / vertexRecoEffcy_ *
                    (hfTowerEtaWidth_ * hfTowerPhiWidth_);

  for (auto const& jet : *theJets) {
    double pt_jet = jet.pt();
    double eta_jet = jet.eta();

    //If central or low pt jets, fill with dummy variables
    if (pt_jet <= jetPtThreshold_ || std::abs(eta_jet) <= jetEtaThreshold_) {
      v_sigmaEtaEta.push_back(-1.);
      v_sigmaPhiPhi.push_back(-1.);
      v_size_CentralEtaStrip.push_back(0);
      v_size_AdjacentEtaStrips.push_back(0);
    } else {
      //First loop over PF candidates to compute some global variables needed for shower shape calculations
      double sumptPFcands = 0.;

      for (unsigned i = 0; i < jet.numberOfSourceCandidatePtrs(); ++i) {
        const reco::Candidate* icand = jet.sourceCandidatePtr(i).get();
        //Only look at pdgId =1,2 (HF PF cands)
        if (std::abs(icand->pdgId()) > 2)
          continue;
        double pt_PUsub = icand->pt() - puoffset;
        if (pt_PUsub < widthPtThreshold_)
          continue;
        sumptPFcands += pt_PUsub;
      }

      //Second loop to compute the various shower shape variables
      int size_CentralEtaStrip(0.), size_AdjacentEtaStrips(0.);
      double sigmaEtaEtaSq(0.), sigmaPhiPhiSq(0.);
      double sumweightsPFcands = 0;
      for (unsigned i = 0; i < jet.numberOfSourceCandidatePtrs(); ++i) {
        const reco::Candidate* icand = jet.sourceCandidatePtr(i).get();
        //Only look at pdgId =1,2 (HF PF cands)
        if (std::abs(icand->pdgId()) > 2)
          continue;

        double deta = std::abs(icand->eta() - jet.eta());
        double dphi = std::abs(deltaPhi(icand->phi(), jet.phi()));
        double pt_PUsub = icand->pt() - puoffset;

        //This is simply the size of the central eta strip and the adjacent strips
        if (pt_PUsub >= stripPtThreshold_) {
          if (dphi <= hfTowerPhiWidth_ * 0.5)
            size_CentralEtaStrip++;
          else if (dphi <= hfTowerPhiWidth_ * 1.5)
            size_AdjacentEtaStrips++;
        }

        //Now computing sigmaEtaEta/PhiPhi
        if (pt_PUsub >= widthPtThreshold_ && sumptPFcands > 0) {
          double weight = pt_PUsub / sumptPFcands;
          sigmaEtaEtaSq += deta * deta * weight;
          sigmaPhiPhiSq += dphi * dphi * weight;
          sumweightsPFcands += weight;
        }
      }

      v_size_CentralEtaStrip.push_back(size_CentralEtaStrip);
      v_size_AdjacentEtaStrips.push_back(size_AdjacentEtaStrips);

      if (sumweightsPFcands > 0 && sigmaEtaEtaSq > 0 && sigmaPhiPhiSq > 0) {
        v_sigmaEtaEta.push_back(sqrt(sigmaEtaEtaSq / sumweightsPFcands));
        v_sigmaPhiPhi.push_back(sqrt(sigmaPhiPhiSq / sumweightsPFcands));
      } else {
        v_sigmaEtaEta.push_back(-1.);
        v_sigmaPhiPhi.push_back(-1.);
      }

    }  //End loop over jets
  }

  putInEvent("sigmaEtaEta", theJets, v_sigmaEtaEta, iEvent);
  putInEvent("sigmaPhiPhi", theJets, v_sigmaPhiPhi, iEvent);
  putInEvent("centralEtaStripSize", theJets, v_size_CentralEtaStrip, iEvent);
  putInEvent("adjacentEtaStripsSize", theJets, v_size_AdjacentEtaStrips, iEvent);
}

/// Function to put product into event
template <typename T>
void HFJetShowerShape::putInEvent(const std::string& name,
                                  const edm::Handle<edm::View<reco::Jet>>& jets,
                                  std::vector<T> product,
                                  edm::Event& iEvent) const {
  auto out = std::make_unique<edm::ValueMap<T>>();
  typename edm::ValueMap<T>::Filler filler(*out);
  filler.insert(jets, product.begin(), product.end());
  filler.fill();
  iEvent.put(std::move(out), name);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HFJetShowerShape::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.add<double>("jetPtThreshold", 25.);
  desc.add<double>("jetEtaThreshold", 2.9);
  desc.add<double>("hfTowerEtaWidth", 0.175);
  desc.add<double>("hfTowerPhiWidth", 0.175);
  desc.add<double>("vertexRecoEffcy", 0.7);
  desc.add<double>("offsetPerPU", 0.4);
  desc.add<double>("jetReferenceRadius", 0.4);
  desc.add<double>("stripPtThreshold", 10.);
  desc.add<double>("widthPtThreshold", 3.);
  descriptions.add("hfJetShowerShape", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HFJetShowerShape);
