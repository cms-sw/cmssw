#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <map>

using namespace l1t;

class L1GTNTupleProducer : public edm::one::EDAnalyzer<> {
public:
  explicit L1GTNTupleProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  //void beginJob(void) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  //void endJob() override;

private:
  void fillData(const l1t::P2GTCandidateCollection& collection, const std::string& instanceName);

  const std::string producerName_;
  const unsigned maxNTuples_;

  const edm::EDGetTokenT<P2GTCandidateCollection> gttPromptJetToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gttDisplacedJetToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gttPrimaryVertexToken_;

  const edm::EDGetTokenT<P2GTCandidateCollection> gmtSaPromptMuonToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gmtSaDisplacedMuonToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gmtTkMuonToken_;

  const edm::EDGetTokenT<P2GTCandidateCollection> cl2JetToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> cl2PhotonToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> cl2ElectronToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> cl2EtSumToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> cl2HtSumToken_;

  const edm::Service<TFileService> fs_;
  TTree* tree_;

  std::unordered_map<std::string, std::vector<int>> data_;
};

static const std::map<std::string, std::vector<std::string>> NTUPLE_VARIABLES = {
    {"GTTPromptJet", {"Pt", "Eta", "Phi"}},
    {"GTTDisplacedJet", {"Pt", "Eta", "Phi"}},
    {"GTTPrimaryVertex", {"Z0"}},
    {"GMTSaPromptMuon", {"Pt", "Eta", "Phi", "Z0"}},
    {"GMTSaDisplacedMuon", {"Pt", "Eta", "Phi", "Z0"}},
    {"GMTTkMuon", {"Pt", "Eta", "Phi", "Z0"}},
    {"CL2Jet", {"Pt", "Eta", "Phi", "Z0"}},
    {"CL2Photon", {"Pt", "Eta", "Phi"}},
    {"CL2Electron", {"Pt", "Eta", "Phi", "Z0"}},
    {"CL2EtSum", {"Pt", "Phi", "ScaSumPt"}},
    {"CL2HtSum", {"Pt", "Phi", "ScaSumPt"}}};

L1GTNTupleProducer::L1GTNTupleProducer(const edm::ParameterSet& config)
    : producerName_(config.getParameter<std::string>("producerName")),
      maxNTuples_(config.getParameter<unsigned>("maxNTuples")),
      gttPromptJetToken_(consumes<P2GTCandidateCollection>(edm::InputTag(producerName_, "GTTPromptJets"))),
      gttDisplacedJetToken_(consumes<P2GTCandidateCollection>(edm::InputTag(producerName_, "GTTDisplacedJets"))),
      gttPrimaryVertexToken_(consumes<P2GTCandidateCollection>(edm::InputTag(producerName_, "GTTPrimaryVert"))),
      gmtSaPromptMuonToken_(consumes<P2GTCandidateCollection>(edm::InputTag(producerName_, "GMTSaPromptMuons"))),
      gmtSaDisplacedMuonToken_(consumes<P2GTCandidateCollection>(edm::InputTag(producerName_, "GMTSaDisplacedMuons"))),
      gmtTkMuonToken_(consumes<P2GTCandidateCollection>(edm::InputTag(producerName_, "GMTTkMuons"))),
      cl2JetToken_(consumes<P2GTCandidateCollection>(edm::InputTag(producerName_, "CL2Jets"))),
      cl2PhotonToken_(consumes<P2GTCandidateCollection>(edm::InputTag(producerName_, "CL2Photons"))),
      cl2ElectronToken_(consumes<P2GTCandidateCollection>(edm::InputTag(producerName_, "CL2Electrons"))),
      cl2EtSumToken_(consumes<P2GTCandidateCollection>(edm::InputTag(producerName_, "CL2EtSum"))),
      cl2HtSumToken_(consumes<P2GTCandidateCollection>(edm::InputTag(producerName_, "CL2HtSum"))),
      tree_(fs_->make<TTree>("L1PhaseIITree", "L1PhaseIITree")) {
  for (const auto& [collec, variables] : NTUPLE_VARIABLES) {
    for (const std::string& var : variables) {
      std::string name = collec + var;
      tree_->Branch(name.c_str(), &data_[name], 8000, 1);
    }
  }
}

void L1GTNTupleProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("producerName");
  desc.add<unsigned>("maxNTuples");

  descriptions.addDefault(desc);
}

void L1GTNTupleProducer::fillData(const l1t::P2GTCandidateCollection& collection, const std::string& instanceName) {
  for (const P2GTCandidate& object : collection) {
    for (const std::string& var : NTUPLE_VARIABLES.at(instanceName)) {
      std::string name = instanceName + var;
      if (var == "Pt")
        data_.at(name).push_back(object.hwPT());
      else if (var == "Eta")
        data_.at(name).push_back(object.hwEta());
      else if (var == "Phi")
        data_.at(name).push_back(object.hwPhi());
      else if (var == "Z0")
        data_.at(name).push_back(object.hwZ0());
      else if (var == "D0")
        data_.at(name).push_back(object.hwD0());
      else if (var == "ScaSumPt")
        data_.at(name).push_back(object.hwSca_sum());
    }
  }
}

void L1GTNTupleProducer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  const l1t::P2GTCandidateCollection& gttPromptJets = event.get(gttPromptJetToken_);
  fillData(gttPromptJets, "GTTPromptJet");

  const l1t::P2GTCandidateCollection& gttDisplacedJets = event.get(gttDisplacedJetToken_);
  fillData(gttDisplacedJets, "GTTDisplacedJet");

  const l1t::P2GTCandidateCollection& gttPrimaryVertices = event.get(gttPrimaryVertexToken_);
  fillData(gttPrimaryVertices, "GTTPrimaryVertex");

  const l1t::P2GTCandidateCollection& gmtSaPromptMuons = event.get(gmtSaPromptMuonToken_);
  fillData(gmtSaPromptMuons, "GMTSaPromptMuon");

  const l1t::P2GTCandidateCollection& gmtSaDisplacedMuons = event.get(gmtSaDisplacedMuonToken_);
  fillData(gmtSaDisplacedMuons, "GMTSaDisplacedMuon");

  const l1t::P2GTCandidateCollection& gmtTkMuons = event.get(gmtTkMuonToken_);
  fillData(gmtTkMuons, "GMTTkMuon");

  const l1t::P2GTCandidateCollection& cl2Jets = event.get(cl2JetToken_);
  fillData(cl2Jets, "CL2Jet");

  const l1t::P2GTCandidateCollection& cl2Photons = event.get(cl2PhotonToken_);
  fillData(cl2Photons, "CL2Photon");

  const l1t::P2GTCandidateCollection& cl2Electrons = event.get(cl2ElectronToken_);
  fillData(cl2Electrons, "CL2Electron");

  const l1t::P2GTCandidateCollection& cl2EtSum = event.get(cl2EtSumToken_);
  fillData(cl2EtSum, "CL2EtSum");

  const l1t::P2GTCandidateCollection& cl2HtSum = event.get(cl2HtSumToken_);
  fillData(cl2HtSum, "CL2HtSum");

  tree_->Fill();

  for (auto& [key, vector] : data_) {
    vector.clear();
  }
}

DEFINE_FWK_MODULE(L1GTNTupleProducer);
