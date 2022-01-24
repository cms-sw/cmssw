// -*- C++ -*-
//
// Package:    ClusterAnalyzer
// Class:      ClusterAnalyzer
//
/**\class ClusterAnalyzer ClusterAnalyzer.cc msegala/ClusterSummary/src/ClusterAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Michael Segala
//         Created:  Wed Feb 23 17:36:23 CST 2011
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/TrackerCommon/interface/ClusterSummary.h"

// ROOT includes
#include "TTree.h"
#include "TH1.h"
#include "TFile.h"

//
// class declaration
//

class ClusterSummary;

using namespace std;

class ClusterAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ClusterAnalyzer(const edm::ParameterSet&);
  ~ClusterAnalyzer();

private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<ClusterSummary> token;

  std::map<int, std::string> enumModules_;
  std::map<std::string, TH1D*> histos1D_;

  std::vector<int> nType_;
  std::vector<float> clusterSize_;
  std::vector<float> clusterCharge_;

  std::map<int, std::string> allModules_;

  edm::Service<TFileService> fs;

  bool _verbose;
};

ClusterAnalyzer::ClusterAnalyzer(const edm::ParameterSet& iConfig) {
  usesResource(TFileService::kSharedResource);
  token = consumes<ClusterSummary>(iConfig.getParameter<edm::InputTag>("clusterSum"));
  _verbose = true;  //set to true to see the event by event summary info

  std::vector<std::string> wantedsubdets = iConfig.getParameter<std::vector<std::string> >("wantedSubDets");
  for (auto iS : wantedsubdets) {
    ClusterSummary::CMSTracker subdet = ClusterSummary::NVALIDENUMS;
    for (int iN = 0; iN < ClusterSummary::NVALIDENUMS; ++iN)
      if (ClusterSummary::subDetNames[iN] == iS)
        subdet = ClusterSummary::CMSTracker(iN);
    if (subdet == ClusterSummary::NVALIDENUMS)
      throw cms::Exception("No standard selection: ") << iS;

    allModules_[subdet] = iS;
  }

  std::vector<edm::ParameterSet> wantedusersubdets_ps =
      iConfig.getParameter<std::vector<edm::ParameterSet> >("wantedUserSubDets");
  for (const auto& iS : wantedusersubdets_ps) {
    ClusterSummary::CMSTracker subdet = (ClusterSummary::CMSTracker)iS.getParameter<unsigned int>("detSelection");
    std::string detname = iS.getParameter<std::string>("detLabel");
    if (subdet <= ClusterSummary::NVALIDENUMS)
      throw cms::Exception("Already predefined selection: ") << subdet;
    if (subdet >= ClusterSummary::NTRACKERENUMS)
      throw cms::Exception("Selection is out of range: ") << subdet;
    allModules_[subdet] = detname;
  }

  edm::LogPrint("ClusterAnalyzer") << "From provenance infomation the selected modules are = ";
  for (auto i : allModules_)
    edm::LogPrint("ClusterAnalyzer") << i.second << " ";
  edm::LogPrint("ClusterAnalyzer") << endl;

  for (auto i = allModules_.begin(); i != allModules_.end(); ++i) {
    std::string tmpstr = i->second;
    histos1D_[(tmpstr + "nclusters").c_str()] =
        fs->make<TH1D>((tmpstr + "nclusters").c_str(), (tmpstr + "nclusters").c_str(), 1000, 0, 3000);
    histos1D_[(tmpstr + "nclusters").c_str()]->SetXTitle(("number of Clusters in " + tmpstr).c_str());

    histos1D_[(tmpstr + "avgCharge").c_str()] =
        fs->make<TH1D>((tmpstr + "avgCharge").c_str(), (tmpstr + "avgCharge").c_str(), 500, 0, 1000);
    histos1D_[(tmpstr + "avgCharge").c_str()]->SetXTitle(("average cluster charge in " + tmpstr).c_str());

    histos1D_[(tmpstr + "avgSize").c_str()] =
        fs->make<TH1D>((tmpstr + "avgSize").c_str(), (tmpstr + "avgSize").c_str(), 30, 0, 10);
    histos1D_[(tmpstr + "avgSize").c_str()]->SetXTitle(("average cluster size in " + tmpstr).c_str());
  }
}

ClusterAnalyzer::~ClusterAnalyzer() = default;

void ClusterAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  Handle<ClusterSummary> class_;
  iEvent.getByToken(token, class_);

  for (unsigned int iM = 0; iM < class_->getNumberOfModules(); iM++) {
    auto iAllModules = allModules_.find(class_->getModule(iM));
    if (iAllModules == allModules_.end())
      continue;

    std::string tmpstr = iAllModules->second;

    histos1D_[(tmpstr + "nclusters").c_str()]->Fill(class_->getNClusByIndex(iM));
    histos1D_[(tmpstr + "avgSize").c_str()]->Fill(class_->getClusSizeByIndex(iM) / class_->getNClusByIndex(iM));
    histos1D_[(tmpstr + "avgCharge").c_str()]->Fill(class_->getClusChargeByIndex(iM) / class_->getNClusByIndex(iM));

    edm::LogPrint("ClusterAnalyzer") << "n" << tmpstr << ", avg size, avg charge = " << class_->getNClusByIndex(iM);
    edm::LogPrint("ClusterAnalyzer") << ", " << class_->getClusSizeByIndex(iM) / class_->getNClusByIndex(iM);
    edm::LogPrint("ClusterAnalyzer") << ", " << class_->getClusChargeByIndex(iM) / class_->getNClusByIndex(iM) << endl;
  }
  edm::LogPrint("ClusterAnalyzer") << "-------------------------------------------------------" << endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ClusterAnalyzer);
