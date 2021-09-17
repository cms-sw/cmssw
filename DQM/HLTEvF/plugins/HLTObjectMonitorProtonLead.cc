// -*- C++ -*-
//
// Package:    QM/HLTObjectMonitorProtonLead
// Class:      HLTObjectMonitorProtonLead
//
/**\class HLTObjectMonitorProtonLead HLTObjectMonitorProtonLead.cc DQM/HLTEvF/plugins/HLTObjectMonitorProtonLead.cc 

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Varun Sharma
//         Created:  Fri, 21 Oct 2016 12:29:00 GMT
//
//

// system include files
#include <memory>
#include <sys/time.h>
#include <cstdlib>

// user include files
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//for collections
#include "HLTrigger/JetMET/interface/AlphaT.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "TMath.h"
#include "TStyle.h"
#include "TLorentzVector.h"

#include <unordered_map>
//
// class declaration
//

//using namespace edm;
using namespace trigger;
using std::string;
using std::unordered_map;
using std::vector;

class HLTObjectMonitorProtonLead : public DQMEDAnalyzer {
  struct hltPlot {
    MonitorElement* ME;
    string pathName;
    string pathNameOR;
    string moduleName;
    string moduleNameOR;
    int pathIndex = -99;
    int pathIndexOR = -99;
    string plotLabel;
    string xAxisLabel;
    int nBins;
    double xMin;
    double xMax;
    bool displayInPrimary;
  };

public:
  explicit HLTObjectMonitorProtonLead(const edm::ParameterSet&);
  ~HLTObjectMonitorProtonLead() override;

  //      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker& i, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
  vector<hltPlot*> plotList;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  double get_wall_time(void);
  // ----------member data ---------------------------

  bool debugPrint;
  HLTConfigProvider hltConfig_;
  string topDirectoryName;
  string mainShifterFolder;
  string backupFolder;
  unordered_map<string, bool> acceptMap;
  unordered_map<hltPlot*, edm::ParameterSet*> plotMap;

  //set Token(-s)
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> aodTriggerToken_;

  //declare params
  //pPb run
  edm::ParameterSet caloAK4JetPt_pset;
  edm::ParameterSet caloAK4JetEta_pset;
  edm::ParameterSet caloAK4JetPhi_pset;
  edm::ParameterSet pfAK4JetPt_pset;
  edm::ParameterSet pfAK4JetEta_pset;
  edm::ParameterSet pfAK4JetPhi_pset;
  edm::ParameterSet caloDiAK4JetPt_pset;
  edm::ParameterSet caloDiAK4JetEta_pset;
  edm::ParameterSet caloDiAK4JetPhi_pset;
  edm::ParameterSet pfDiAK4JetPt_pset;
  edm::ParameterSet pfDiAK4JetEta_pset;
  edm::ParameterSet pfDiAK4JetPhi_pset;
  edm::ParameterSet photonPt_pset;
  edm::ParameterSet photonEta_pset;
  edm::ParameterSet photonPhi_pset;
  edm::ParameterSet photonPPPt_pset;
  edm::ParameterSet photonPPEta_pset;
  edm::ParameterSet photonPPPhi_pset;
  edm::ParameterSet caloBJetPt_pset;
  edm::ParameterSet caloBJetEta_pset;
  edm::ParameterSet caloBJetPhi_pset;
  edm::ParameterSet pfBJetPt_pset;
  edm::ParameterSet pfBJetEta_pset;
  edm::ParameterSet pfBJetPhi_pset;
  edm::ParameterSet electronPt_pset;
  edm::ParameterSet electronEta_pset;
  edm::ParameterSet electronPhi_pset;
  edm::ParameterSet l3muon3Pt_pset;
  edm::ParameterSet l3muon3Eta_pset;
  edm::ParameterSet l3muon3Phi_pset;
  edm::ParameterSet l2muon12Pt_pset;
  edm::ParameterSet l2muon12Eta_pset;
  edm::ParameterSet l2muon12Phi_pset;
  edm::ParameterSet l3muon12Pt_pset;
  edm::ParameterSet l3muon12Eta_pset;
  edm::ParameterSet l3muon12Phi_pset;
  edm::ParameterSet pAL1DoubleMuZMass_pset;
  edm::ParameterSet pAL2DoubleMuZMass_pset;
  edm::ParameterSet pAL3DoubleMuZMass_pset;
  edm::ParameterSet wallTime_pset;

  string processName_;

  hltPlot caloAK4JetPt_;
  hltPlot caloAK4JetEta_;
  hltPlot caloAK4JetPhi_;
  hltPlot pfAK4JetPt_;
  hltPlot pfAK4JetEta_;
  hltPlot pfAK4JetPhi_;
  hltPlot caloDiAK4JetPt_;
  hltPlot caloDiAK4JetEta_;
  hltPlot caloDiAK4JetPhi_;
  hltPlot pfDiAK4JetPt_;
  hltPlot pfDiAK4JetEta_;
  hltPlot pfDiAK4JetPhi_;
  hltPlot photonPt_;
  hltPlot photonEta_;
  hltPlot photonPhi_;
  hltPlot photonPPPt_;
  hltPlot photonPPEta_;
  hltPlot photonPPPhi_;
  hltPlot caloBJetPt_;
  hltPlot caloBJetEta_;
  hltPlot caloBJetPhi_;
  hltPlot pfBJetPt_;
  hltPlot pfBJetEta_;
  hltPlot pfBJetPhi_;
  hltPlot electronPt_;
  hltPlot electronEta_;
  hltPlot electronPhi_;
  hltPlot l3muon3Pt_;
  hltPlot l3muon3Eta_;
  hltPlot l3muon3Phi_;
  hltPlot l2muon12Pt_;
  hltPlot l2muon12Eta_;
  hltPlot l2muon12Phi_;
  hltPlot l3muon12Pt_;
  hltPlot l3muon12Eta_;
  hltPlot l3muon12Phi_;
  hltPlot pAL1DoubleMuZMass_;
  hltPlot pAL2DoubleMuZMass_;
  hltPlot pAL3DoubleMuZMass_;
  hltPlot wallTime_;
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
HLTObjectMonitorProtonLead::HLTObjectMonitorProtonLead(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed
  debugPrint = false;

  topDirectoryName = "HLT/ObjectMonitor";
  mainShifterFolder = topDirectoryName + "/MainShifter";
  backupFolder = topDirectoryName + "/Backup";

  //parse params
  processName_ = iConfig.getParameter<string>("processName");

  caloAK4JetPt_pset = iConfig.getParameter<edm::ParameterSet>("caloAK4JetPt");
  plotMap[&caloAK4JetPt_] = &caloAK4JetPt_pset;
  caloAK4JetEta_pset = iConfig.getParameter<edm::ParameterSet>("caloAK4JetEta");
  plotMap[&caloAK4JetEta_] = &caloAK4JetEta_pset;
  caloAK4JetPhi_pset = iConfig.getParameter<edm::ParameterSet>("caloAK4JetPhi");
  plotMap[&caloAK4JetPhi_] = &caloAK4JetPhi_pset;
  pfAK4JetPt_pset = iConfig.getParameter<edm::ParameterSet>("pfAK4JetPt");
  plotMap[&pfAK4JetPt_] = &pfAK4JetPt_pset;
  pfAK4JetEta_pset = iConfig.getParameter<edm::ParameterSet>("pfAK4JetEta");
  plotMap[&pfAK4JetEta_] = &pfAK4JetEta_pset;
  pfAK4JetPhi_pset = iConfig.getParameter<edm::ParameterSet>("pfAK4JetPhi");
  plotMap[&pfAK4JetPhi_] = &pfAK4JetPhi_pset;
  caloDiAK4JetPt_pset = iConfig.getParameter<edm::ParameterSet>("caloDiAK4JetPt");
  plotMap[&caloDiAK4JetPt_] = &caloDiAK4JetPt_pset;
  caloDiAK4JetEta_pset = iConfig.getParameter<edm::ParameterSet>("caloDiAK4JetEta");
  plotMap[&caloDiAK4JetEta_] = &caloDiAK4JetEta_pset;
  caloDiAK4JetPhi_pset = iConfig.getParameter<edm::ParameterSet>("caloDiAK4JetPhi");
  plotMap[&caloDiAK4JetPhi_] = &caloDiAK4JetPhi_pset;
  pfDiAK4JetPt_pset = iConfig.getParameter<edm::ParameterSet>("pfDiAK4JetPt");
  plotMap[&pfDiAK4JetPt_] = &pfDiAK4JetPt_pset;
  pfDiAK4JetEta_pset = iConfig.getParameter<edm::ParameterSet>("pfDiAK4JetEta");
  plotMap[&pfDiAK4JetEta_] = &pfDiAK4JetEta_pset;
  pfDiAK4JetPhi_pset = iConfig.getParameter<edm::ParameterSet>("pfDiAK4JetPhi");
  plotMap[&pfDiAK4JetPhi_] = &pfDiAK4JetPhi_pset;
  photonPt_pset = iConfig.getParameter<edm::ParameterSet>("photonPt");
  plotMap[&photonPt_] = &photonPt_pset;
  photonEta_pset = iConfig.getParameter<edm::ParameterSet>("photonEta");
  plotMap[&photonEta_] = &photonEta_pset;
  photonPhi_pset = iConfig.getParameter<edm::ParameterSet>("photonPhi");
  plotMap[&photonPhi_] = &photonPhi_pset;
  photonPPPt_pset = iConfig.getParameter<edm::ParameterSet>("photonPPPt");
  plotMap[&photonPPPt_] = &photonPPPt_pset;
  photonPPEta_pset = iConfig.getParameter<edm::ParameterSet>("photonPPEta");
  plotMap[&photonPPEta_] = &photonPPEta_pset;
  photonPPPhi_pset = iConfig.getParameter<edm::ParameterSet>("photonPPPhi");
  plotMap[&photonPPPhi_] = &photonPPPhi_pset;
  caloBJetPt_pset = iConfig.getParameter<edm::ParameterSet>("caloBJetPt");
  plotMap[&caloBJetPt_] = &caloBJetPt_pset;
  caloBJetEta_pset = iConfig.getParameter<edm::ParameterSet>("caloBJetEta");
  plotMap[&caloBJetEta_] = &caloBJetEta_pset;
  caloBJetPhi_pset = iConfig.getParameter<edm::ParameterSet>("caloBJetPhi");
  plotMap[&caloBJetPhi_] = &caloBJetPhi_pset;
  pfBJetPt_pset = iConfig.getParameter<edm::ParameterSet>("pfBJetPt");
  plotMap[&pfBJetPt_] = &pfBJetPt_pset;
  pfBJetEta_pset = iConfig.getParameter<edm::ParameterSet>("pfBJetEta");
  plotMap[&pfBJetEta_] = &pfBJetEta_pset;
  pfBJetPhi_pset = iConfig.getParameter<edm::ParameterSet>("pfBJetPhi");
  plotMap[&pfBJetPhi_] = &pfBJetPhi_pset;
  electronPt_pset = iConfig.getParameter<edm::ParameterSet>("electronPt");
  plotMap[&electronPt_] = &electronPt_pset;
  electronEta_pset = iConfig.getParameter<edm::ParameterSet>("electronEta");
  plotMap[&electronEta_] = &electronEta_pset;
  electronPhi_pset = iConfig.getParameter<edm::ParameterSet>("electronPhi");
  plotMap[&electronPhi_] = &electronPhi_pset;
  l3muon3Pt_pset = iConfig.getParameter<edm::ParameterSet>("l3muon3Pt");
  plotMap[&l3muon3Pt_] = &l3muon3Pt_pset;
  l3muon3Eta_pset = iConfig.getParameter<edm::ParameterSet>("l3muon3Eta");
  plotMap[&l3muon3Eta_] = &l3muon3Eta_pset;
  l3muon3Phi_pset = iConfig.getParameter<edm::ParameterSet>("l3muon3Phi");
  plotMap[&l3muon3Phi_] = &l3muon3Phi_pset;
  l2muon12Pt_pset = iConfig.getParameter<edm::ParameterSet>("l2muon12Pt");
  plotMap[&l2muon12Pt_] = &l2muon12Pt_pset;
  l2muon12Eta_pset = iConfig.getParameter<edm::ParameterSet>("l2muon12Eta");
  plotMap[&l2muon12Eta_] = &l2muon12Eta_pset;
  l2muon12Phi_pset = iConfig.getParameter<edm::ParameterSet>("l2muon12Phi");
  plotMap[&l2muon12Phi_] = &l2muon12Phi_pset;
  l3muon12Pt_pset = iConfig.getParameter<edm::ParameterSet>("l3muon12Pt");
  plotMap[&l3muon12Pt_] = &l3muon12Pt_pset;
  l3muon12Eta_pset = iConfig.getParameter<edm::ParameterSet>("l3muon12Eta");
  plotMap[&l3muon12Eta_] = &l3muon12Eta_pset;
  l3muon12Phi_pset = iConfig.getParameter<edm::ParameterSet>("l3muon12Phi");
  plotMap[&l3muon12Phi_] = &l3muon12Phi_pset;
  pAL1DoubleMuZMass_pset = iConfig.getParameter<edm::ParameterSet>("pAL1DoubleMuZMass");
  plotMap[&pAL1DoubleMuZMass_] = &pAL1DoubleMuZMass_pset;
  pAL2DoubleMuZMass_pset = iConfig.getParameter<edm::ParameterSet>("pAL2DoubleMuZMass");
  plotMap[&pAL2DoubleMuZMass_] = &pAL2DoubleMuZMass_pset;
  pAL3DoubleMuZMass_pset = iConfig.getParameter<edm::ParameterSet>("pAL3DoubleMuZMass");
  plotMap[&pAL3DoubleMuZMass_] = &pAL3DoubleMuZMass_pset;
  wallTime_pset = iConfig.getParameter<edm::ParameterSet>("wallTime");
  plotMap[&wallTime_] = &wallTime_pset;

  for (auto item = plotMap.begin(); item != plotMap.end(); item++) {
    (*item->first).pathName = (*item->second).getParameter<string>("pathName");
    (*item->first).moduleName = (*item->second).getParameter<string>("moduleName");
    (*item->first).nBins = (*item->second).getParameter<int>("NbinsX");
    (*item->first).xMin = (*item->second).getParameter<double>("Xmin");
    (*item->first).xMax = (*item->second).getParameter<double>("Xmax");
    (*item->first).xAxisLabel = (*item->second).getParameter<string>("axisLabel");
    (*item->first).plotLabel = (*item->second).getParameter<string>("plotLabel");
    (*item->first).displayInPrimary = (*item->second).getParameter<bool>("mainWorkspace");

    if ((*item->second).exists("pathName_OR")) {
      (*item->first).pathNameOR = (*item->second).getParameter<string>("pathName_OR");
    }
    if ((*item->second).exists("moduleName_OR")) {
      (*item->first).moduleNameOR = (*item->second).getParameter<string>("moduleName_OR");
    }

    plotList.push_back(item->first);
  }
  plotMap.clear();

  //set Token(s)
  triggerResultsToken_ = consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("triggerResults"));
  aodTriggerToken_ = consumes<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag>("triggerEvent"));
}

HLTObjectMonitorProtonLead::~HLTObjectMonitorProtonLead() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void HLTObjectMonitorProtonLead::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  double start = get_wall_time();

  using namespace edm;

  if (debugPrint)
    std::cout << "Inside analyze(). " << std::endl;

  // access trigger results
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
  if (!triggerResults.isValid())
    return;

  edm::Handle<trigger::TriggerEvent> aodTriggerEvent;
  iEvent.getByToken(aodTriggerToken_, aodTriggerEvent);
  if (!aodTriggerEvent.isValid())
    return;

  //reset everything to not accepted at beginning of each event
  unordered_map<string, bool> firedMap = acceptMap;
  for (auto plot : plotList)  //loop over paths
  {
    if (firedMap[plot->pathName])
      continue;
    bool triggerAccept = false;
    const TriggerObjectCollection objects = aodTriggerEvent->getObjects();
    edm::InputTag moduleFilter;
    std::string pathName;
    if (plot->pathIndex > 0 && triggerResults->accept(plot->pathIndex) && hltConfig_.saveTags(plot->moduleName)) {
      moduleFilter = edm::InputTag(plot->moduleName, "", processName_);
      pathName = plot->pathName;
      triggerAccept = true;
    } else if (plot->pathIndexOR > 0 && triggerResults->accept(plot->pathIndexOR) &&
               hltConfig_.saveTags(plot->moduleNameOR)) {
      if (firedMap[plot->pathNameOR])
        continue;
      moduleFilter = edm::InputTag(plot->moduleNameOR, "", processName_);
      pathName = plot->pathNameOR;
      triggerAccept = true;
    }

    if (triggerAccept) {
      unsigned int moduleFilterIndex = aodTriggerEvent->filterIndex(moduleFilter);

      if (moduleFilterIndex + 1 > aodTriggerEvent->sizeFilters())
        return;
      const Keys& keys = aodTriggerEvent->filterKeys(moduleFilterIndex);

      ////////////////////////////////
      ///
      /// single-object plots
      ///
      ////////////////////////////////

      //calo AK4 jet pt + eta + phi
      if (pathName == caloAK4JetPt_.pathName) {
        for (const auto& key : keys) {
          caloAK4JetPt_.ME->Fill(objects[key].pt());
          caloAK4JetEta_.ME->Fill(objects[key].eta());
          caloAK4JetPhi_.ME->Fill(objects[key].phi());
        }
      }
      //pf AK4 jet pt + eta + phi
      else if (pathName == pfAK4JetPt_.pathName) {
        for (const auto& key : keys) {
          pfAK4JetPt_.ME->Fill(objects[key].pt());
          pfAK4JetEta_.ME->Fill(objects[key].eta());
          pfAK4JetPhi_.ME->Fill(objects[key].phi());
        }
      }
      //calo AK4 di jet pt + eta + phi
      else if (pathName == caloDiAK4JetPt_.pathName) {
        for (const auto& key : keys) {
          caloDiAK4JetPt_.ME->Fill(objects[key].pt());
          caloDiAK4JetEta_.ME->Fill(objects[key].eta());
          caloDiAK4JetPhi_.ME->Fill(objects[key].phi());
        }
      }
      //pf AK4 di jet pt + eta + phi
      else if (pathName == pfDiAK4JetPt_.pathName) {
        for (const auto& key : keys) {
          pfDiAK4JetPt_.ME->Fill(objects[key].pt());
          pfDiAK4JetEta_.ME->Fill(objects[key].eta());
          pfDiAK4JetPhi_.ME->Fill(objects[key].phi());
        }
      }
      //photon pt + eta + phi
      else if (pathName == photonPt_.pathName) {
        for (const auto& key : keys) {
          photonPt_.ME->Fill(objects[key].pt());
          photonEta_.ME->Fill(objects[key].eta());
          photonPhi_.ME->Fill(objects[key].phi());
        }
      }
      //photon PPStyle pt + eta + phi
      else if (pathName == photonPPPt_.pathName) {
        for (const auto& key : keys) {
          photonPPPt_.ME->Fill(objects[key].pt());
          photonPPEta_.ME->Fill(objects[key].eta());
          photonPPPhi_.ME->Fill(objects[key].phi());
        }
      }
      //calo Bjet pt + eta + phi
      else if (pathName == caloBJetPt_.pathName) {
        for (const auto& key : keys) {
          caloBJetPt_.ME->Fill(objects[key].pt());
          caloBJetEta_.ME->Fill(objects[key].eta());
          caloBJetPhi_.ME->Fill(objects[key].phi());
        }
      }
      //PF Bjet pt + eta + phi
      else if (pathName == pfBJetPt_.pathName) {
        for (const auto& key : keys) {
          pfBJetPt_.ME->Fill(objects[key].pt());
          pfBJetEta_.ME->Fill(objects[key].eta());
          pfBJetPhi_.ME->Fill(objects[key].phi());
        }
      }
      //electron pt + eta + phi
      else if (pathName == electronPt_.pathName) {
        for (const auto& key : keys) {
          electronPt_.ME->Fill(objects[key].pt());
          electronEta_.ME->Fill(objects[key].eta());
          electronPhi_.ME->Fill(objects[key].phi());
        }
      } else if (pathName == l3muon3Pt_.pathName) {
        for (const auto& key : keys) {
          l3muon3Pt_.ME->Fill(objects[key].pt());
          l3muon3Eta_.ME->Fill(objects[key].eta());
          l3muon3Phi_.ME->Fill(objects[key].phi());
        }
      } else if (pathName == l2muon12Pt_.pathName) {
        for (const auto& key : keys) {
          l2muon12Pt_.ME->Fill(objects[key].pt());
          l2muon12Eta_.ME->Fill(objects[key].eta());
          l2muon12Phi_.ME->Fill(objects[key].phi());
        }
      } else if (pathName == l3muon12Pt_.pathName) {
        for (const auto& key : keys) {
          l3muon12Pt_.ME->Fill(objects[key].pt());
          l3muon12Eta_.ME->Fill(objects[key].eta());
          l3muon12Phi_.ME->Fill(objects[key].phi());
        }
      }

      // ////////////////////////////////
      // ///
      // /// double-object plots
      // ///
      // ////////////////////////////////

      else if (pathName == pAL1DoubleMuZMass_.pathName) {
        const double mu_mass(.105658);
        unsigned int kCnt0 = 0;
        for (const auto& key0 : keys) {
          unsigned int kCnt1 = 0;
          for (const auto& key1 : keys) {
            if (key0 != key1 &&
                kCnt1 > kCnt0) {  // avoid filling hists with same objs && avoid double counting separate objs
              // if (abs(objects[key0].id()) == 13 && (objects[key0].id()+objects[key1].id()==0))  // id is not filled for l1 stage2 muons
              TLorentzVector mu1, mu2, dimu;
              mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
              mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
              dimu = mu1 + mu2;
              if (dimu.M() > pAL1DoubleMuZMass_.xMin && dimu.M() < pAL1DoubleMuZMass_.xMax)
                pAL1DoubleMuZMass_.ME->Fill(dimu.M());
            }
            kCnt1 += 1;
          }
          kCnt0 += 1;
        }
      } else if (pathName == pAL2DoubleMuZMass_.pathName) {
        const double mu_mass(.105658);
        unsigned int kCnt0 = 0;
        for (const auto& key0 : keys) {
          unsigned int kCnt1 = 0;
          for (const auto& key1 : keys) {
            if (key0 != key1 &&
                kCnt1 > kCnt0) {  // avoid filling hists with same objs && avoid double counting separate objs
              if (abs(objects[key0].id()) == 13 &&
                  (objects[key0].id() + objects[key1].id() == 0)) {  // check muon id and dimuon charge
                TLorentzVector mu1, mu2, dimu;
                mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
                mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
                dimu = mu1 + mu2;
                if (dimu.M() > pAL2DoubleMuZMass_.xMin && dimu.M() < pAL2DoubleMuZMass_.xMax)
                  pAL2DoubleMuZMass_.ME->Fill(dimu.M());
              }
            }
            kCnt1 += 1;
          }
          kCnt0 += 1;
        }
      } else if (pathName == pAL3DoubleMuZMass_.pathName) {
        const double mu_mass(.105658);
        unsigned int kCnt0 = 0;
        for (const auto& key0 : keys) {
          unsigned int kCnt1 = 0;
          for (const auto& key1 : keys) {
            if (key0 != key1 &&
                kCnt1 > kCnt0) {  // avoid filling hists with same objs && avoid double counting separate objs
              if (abs(objects[key0].id()) == 13 &&
                  (objects[key0].id() + objects[key1].id() == 0)) {  // check muon id and dimuon charge
                TLorentzVector mu1, mu2, dimu;
                mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
                mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
                dimu = mu1 + mu2;
                if (dimu.M() > pAL3DoubleMuZMass_.xMin && dimu.M() < pAL3DoubleMuZMass_.xMax)
                  pAL3DoubleMuZMass_.ME->Fill(dimu.M());
              }
            }
            kCnt1 += 1;
          }
          kCnt0 += 1;
        }
      }

      firedMap[pathName] = true;
    }  //end if trigger accept
  }    //end loop over plots/paths

  //   sleep(1); //sleep for 1s, used to calibrate timing
  double end = get_wall_time();
  double wallTime = end - start;
  wallTime_.ME->Fill(wallTime);
}

// ------------ method called when starting to processes a run  ------------
void HLTObjectMonitorProtonLead::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  if (debugPrint)
    std::cout << "Calling beginRun. " << std::endl;
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, processName_, changed)) {
    if (debugPrint)
      std::cout << "Extracting HLTconfig. " << std::endl;
  }

  //get path indicies from menu
  string pathName_noVersion;
  vector<string> triggerPaths = hltConfig_.triggerNames();

  for (const auto& pathName : triggerPaths) {
    pathName_noVersion = hltConfig_.removeVersion(pathName);
    for (auto plot : plotList) {
      if (plot->pathName == pathName_noVersion) {
        (*plot).pathIndex = hltConfig_.triggerIndex(pathName);
      } else if (plot->pathNameOR == pathName_noVersion) {
        (*plot).pathIndexOR = hltConfig_.triggerIndex(pathName);
      }
    }
  }
  vector<hltPlot*> plotList_temp;
  for (auto plot : plotList) {
    if (plot->pathIndex > 0 || plot->pathIndexOR > 0) {
      plotList_temp.push_back(plot);
      acceptMap[plot->pathName] = false;
      if (plot->pathIndexOR > 0)
        acceptMap[plot->pathNameOR] = false;
    }
  }
  //now re-assign plotList to contain only the plots with paths in the menu.
  plotList = plotList_temp;
  plotList_temp.clear();
}

// ------------ method called when ending the processing of a run  ------------

void HLTObjectMonitorProtonLead::bookHistograms(DQMStore::IBooker& ibooker,
                                                edm::Run const& iRun,
                                                edm::EventSetup const& iSetup) {
  ////////////////////////////////
  ///
  /// Main shifter workspace plots
  ///
  ////////////////////////////////

  //book wall time separately
  ibooker.setCurrentFolder(mainShifterFolder);
  wallTime_.ME =
      ibooker.book1D(wallTime_.plotLabel, wallTime_.pathName, wallTime_.nBins, wallTime_.xMin, wallTime_.xMax);
  wallTime_.ME->setAxisTitle(wallTime_.xAxisLabel);

  for (auto plot : plotList) {
    std::string display_pathNames = plot->pathName;
    if (!plot->pathNameOR.empty())
      display_pathNames = plot->pathName + " OR " + plot->pathNameOR;

    if (plot->displayInPrimary) {
      ibooker.setCurrentFolder(mainShifterFolder);
      (*plot).ME = ibooker.book1D(plot->plotLabel, display_pathNames.c_str(), plot->nBins, plot->xMin, plot->xMax);
      (*plot).ME->setAxisTitle(plot->xAxisLabel);
      //need to add OR statement
    } else {
      ibooker.setCurrentFolder(backupFolder);
      (*plot).ME = ibooker.book1D(plot->plotLabel, display_pathNames.c_str(), plot->nBins, plot->xMin, plot->xMax);
      (*plot).ME->setAxisTitle(plot->xAxisLabel);
    }
  }
}

double HLTObjectMonitorProtonLead::get_wall_time() {
  struct timeval time;
  if (gettimeofday(&time, nullptr))
    return 0;
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
HLTObjectMonitorProtonLead::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
HLTObjectMonitorProtonLead::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
// void
// HLTObjectMonitorProtonLead::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   //The following says we do not know what parameters are allowed so do no validation
//   // Please change this to state exactly what you do use, even if it is no parameters
//   edm::ParameterSetDescription desc;
//   desc.setUnknown();
//   descriptions.addDefault(desc);
// }

//define this as a plug-in
DEFINE_FWK_MODULE(HLTObjectMonitorProtonLead);
