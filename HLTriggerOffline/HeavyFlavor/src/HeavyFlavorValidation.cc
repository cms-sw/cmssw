/**\class HeavyFlavorValidation HeavyFlavorValidation.cc HLTriggerOfflineHeavyFlavor/src/HeavyFlavorValidation.cc

 Description: Analyzer to fill Monitoring Elements for muon, dimuon and trigger path efficiency studies (HLT/RECO, RECO/GEN)

 Implementation:
     matching is based on closest in delta R, no duplicates allowed. Generated to Global based on momentum at IP; L1, L2, L2v to Global based on position in muon system, L3 to Global based on momentum at IP.
*/
// Original Author:  Zoltan Gecse

#include <memory>
#include <initializer_list>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "CommonTools/Utils/interface/PtComparator.h"

#include "TLorentzVector.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;
using namespace trigger;

class HeavyFlavorValidation : public DQMEDAnalyzer {
public:
  explicit HeavyFlavorValidation(const edm::ParameterSet &);
  ~HeavyFlavorValidation() override;

protected:
  void dqmBeginRun(const edm::Run &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  int getMotherId(const Candidate *p);
  void match(MonitorElement *me,
             vector<LeafCandidate> &from,
             vector<LeafCandidate> &to,
             double deltaRMatchingCut,
             vector<int> &map);
  void myBook2D(DQMStore::IBooker &ibooker,
                TString name,
                vector<double> &xBins,
                TString xLabel,
                vector<double> &yBins,
                TString yLabel,
                TString title);
  void myBook2D(DQMStore::IBooker &ibooker,
                TString name,
                vector<double> &xBins,
                TString xLabel,
                vector<double> &yBins,
                TString yLabel) {
    myBook2D(ibooker, name, xBins, xLabel, yBins, yLabel, name);
  }
  void myBookProfile2D(DQMStore::IBooker &ibooker,
                       TString name,
                       vector<double> &xBins,
                       TString xLabel,
                       vector<double> &yBins,
                       TString yLabel,
                       TString title);
  void myBookProfile2D(DQMStore::IBooker &ibooker,
                       TString name,
                       vector<double> &xBins,
                       TString xLabel,
                       vector<double> &yBins,
                       TString yLabel) {
    myBookProfile2D(ibooker, name, xBins, xLabel, yBins, yLabel, name);
  }
  void myBook1D(DQMStore::IBooker &ibooker, TString name, vector<double> &xBins, TString label, TString title);
  void myBook1D(DQMStore::IBooker &ibooker, TString name, vector<double> &xBins, TString label) {
    myBook1D(ibooker, name, xBins, label, name);
  }

  /**
     * Get the filter "level" (as it is defined for the use of this module and its corresponding
     * harvesting module).
     *
     * level 1 - 3 -> more or less synonymously to the the trigger levels
     * level 4 and 5 -> vertex, dz, track, etc.. filters
     *
     * See the comments in the definition for some more details.
     */
  int getFilterLevel(const std::string &moduleName, const HLTConfigProvider &hltConfig);

  string dqmFolder;
  string triggerProcessName;
  string triggerPathName;

  EDGetTokenT<TriggerEventWithRefs> triggerSummaryRAWTag;
  EDGetTokenT<TriggerEvent> triggerSummaryAODTag;
  InputTag triggerResultsTag;
  EDGetTokenT<TriggerResults> triggerResultsToken;
  InputTag recoMuonsTag;
  EDGetTokenT<MuonCollection> recoMuonsToken;
  InputTag genParticlesTag;
  EDGetTokenT<GenParticleCollection> genParticlesToken;

  vector<int> motherIDs;
  double genGlobDeltaRMatchingCut;
  double globL1DeltaRMatchingCut;
  double globL2DeltaRMatchingCut;
  double globL3DeltaRMatchingCut;
  vector<double> deltaEtaBins;
  vector<double> deltaPhiBins;
  vector<double> muonPtBins;
  vector<double> muonEtaBins;
  vector<double> muonPhiBins;
  vector<double> dimuonPtBins;
  vector<double> dimuonEtaBins;
  vector<double> dimuonDRBins;
  map<TString, MonitorElement *> ME;
  vector<pair<string, int> > filterNamesLevels;
  const double muonMass;
};

HeavyFlavorValidation::HeavyFlavorValidation(const ParameterSet &pset)
    :  //get parameters
      dqmFolder(pset.getUntrackedParameter<string>("DQMFolder")),
      triggerProcessName(pset.getUntrackedParameter<string>("TriggerProcessName")),
      triggerPathName(pset.getUntrackedParameter<string>("TriggerPathName")),
      motherIDs(pset.getUntrackedParameter<vector<int> >("MotherIDs")),
      genGlobDeltaRMatchingCut(pset.getUntrackedParameter<double>("GenGlobDeltaRMatchingCut")),
      globL1DeltaRMatchingCut(pset.getUntrackedParameter<double>("GlobL1DeltaRMatchingCut")),
      globL2DeltaRMatchingCut(pset.getUntrackedParameter<double>("GlobL2DeltaRMatchingCut")),
      globL3DeltaRMatchingCut(pset.getUntrackedParameter<double>("GlobL3DeltaRMatchingCut")),
      deltaEtaBins(pset.getUntrackedParameter<vector<double> >("DeltaEtaBins")),
      deltaPhiBins(pset.getUntrackedParameter<vector<double> >("DeltaPhiBins")),
      muonPtBins(pset.getUntrackedParameter<vector<double> >("MuonPtBins")),
      muonEtaBins(pset.getUntrackedParameter<vector<double> >("MuonEtaBins")),
      muonPhiBins(pset.getUntrackedParameter<vector<double> >("MuonPhiBins")),
      dimuonPtBins(pset.getUntrackedParameter<vector<double> >("DimuonPtBins")),
      dimuonEtaBins(pset.getUntrackedParameter<vector<double> >("DimuonEtaBins")),
      dimuonDRBins(pset.getUntrackedParameter<vector<double> >("DimuonDRBins")),
      muonMass(0.106) {
  triggerSummaryRAWTag = consumes<TriggerEventWithRefs>(
      InputTag(pset.getUntrackedParameter<string>("TriggerSummaryRAW"), "", triggerProcessName));
  triggerSummaryAODTag =
      consumes<TriggerEvent>(InputTag(pset.getUntrackedParameter<string>("TriggerSummaryAOD"), "", triggerProcessName));
  triggerResultsTag = InputTag(pset.getUntrackedParameter<string>("TriggerResults"), "", triggerProcessName);
  triggerResultsToken = consumes<TriggerResults>(triggerResultsTag);
  recoMuonsTag = pset.getParameter<InputTag>("RecoMuons");
  recoMuonsToken = consumes<MuonCollection>(recoMuonsTag);
  genParticlesTag = pset.getParameter<InputTag>("GenParticles");
  genParticlesToken = consumes<GenParticleCollection>(genParticlesTag);
}

void HeavyFlavorValidation::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  //discover HLT configuration
  HLTConfigProvider hltConfig;
  bool isChanged;
  if (hltConfig.init(iRun, iSetup, triggerProcessName, isChanged)) {
    LogDebug("HLTriggerOfflineHeavyFlavor")
        << "Successfully initialized HLTConfigProvider with process name: " << triggerProcessName << endl;
  } else {
    LogWarning("HLTriggerOfflineHeavyFlavor")
        << "Could not initialize HLTConfigProvider with process name: " << triggerProcessName << endl;
    return;
  }
  vector<string> triggerNames = hltConfig.triggerNames();
  for (const auto &trigName : triggerNames) {
    // TString triggerName = trigName;
    if (trigName.find(triggerPathName) != std::string::npos) {
      vector<string> moduleNames = hltConfig.moduleLabels(trigName);
      for (const auto &moduleName : moduleNames) {
        const int level = getFilterLevel(moduleName, hltConfig);
        if (level > 0) {
          filterNamesLevels.push_back({moduleName, level});
        }
      }
      break;
    }
  }

  if (filterNamesLevels.empty()) {
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Bad Trigger Path: " << triggerPathName << endl;
    return;
  } else {
    std::string str;
    str.reserve(
        512);  // avoid too many realloctions in the following loop (allows for filter names with roughly 100 chars each)
    for (const auto &filters : filterNamesLevels)
      str = str + " " + filters.first;
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Trigger Path: " << triggerPathName << " has filters:" << str;
  }
}

void HeavyFlavorValidation::bookHistograms(DQMStore::IBooker &ibooker,
                                           edm::Run const &iRun,
                                           edm::EventSetup const &iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder((dqmFolder + "/") + triggerProcessName + "/" + triggerPathName);

  // create Monitor Elements
  // Eta Pt Single
  myBook2D(ibooker, "genMuon_genEtaPt", muonEtaBins, "#mu eta", muonPtBins, " #mu pT (GeV)");
  myBook2D(ibooker, "globMuon_genEtaPt", muonEtaBins, "#mu eta", muonPtBins, " #mu pT (GeV)");
  myBook2D(ibooker, "globMuon_recoEtaPt", muonEtaBins, "#mu eta", muonPtBins, " #mu pT (GeV)");

  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBook2D(ibooker,
             TString::Format("filt%dMuon_recoEtaPt", int(i + 1)),
             muonEtaBins,
             "#mu eta",
             muonPtBins,
             " #mu pT (GeV)",
             filterNamesLevels[i].first);
  }
  myBook2D(ibooker, "pathMuon_recoEtaPt", muonEtaBins, "#mu eta", muonPtBins, " #mu pT (GeV)", triggerPathName);
  myBook2D(ibooker, "resultMuon_recoEtaPt", muonEtaBins, "#mu eta", muonPtBins, " #mu pT (GeV)");
  // Eta Pt Single Resolution
  myBookProfile2D(ibooker, "resGlobGen_genEtaPt", muonEtaBins, "#mu eta", muonPtBins, " #mu pT (GeV)");
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBookProfile2D(ibooker,
                    TString::Format("resFilt%dGlob_recoEtaPt", int(i + 1)),
                    muonEtaBins,
                    "#mu eta",
                    muonPtBins,
                    " #mu pT (GeV)",
                    filterNamesLevels[i].first);
  }
  myBookProfile2D(
      ibooker, "resPathGlob_recoEtaPt", muonEtaBins, "#mu eta", muonPtBins, " #mu pT (GeV)", triggerPathName);
  // Eta Pt Double
  myBook2D(ibooker, "genDimuon_genEtaPt", dimuonEtaBins, "#mu#mu eta", dimuonPtBins, " #mu#mu pT (GeV)");
  myBook2D(ibooker, "globDimuon_genEtaPt", dimuonEtaBins, "#mu#mu eta", dimuonPtBins, " #mu#mu pT (GeV)");
  myBook2D(ibooker, "globDimuon_recoEtaPt", dimuonEtaBins, "#mu#mu eta", dimuonPtBins, " #mu#mu pT (GeV)");
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBook2D(ibooker,
             TString::Format("filt%dDimuon_recoEtaPt", int(i + 1)),
             dimuonEtaBins,
             "#mu#mu eta",
             dimuonPtBins,
             " #mu#mu pT (GeV)",
             filterNamesLevels[i].first);
  }
  myBook2D(
      ibooker, "pathDimuon_recoEtaPt", dimuonEtaBins, "#mu#mu eta", dimuonPtBins, " #mu#mu pT (GeV)", triggerPathName);
  myBook2D(ibooker, "resultDimuon_recoEtaPt", dimuonEtaBins, "#mu#mu eta", dimuonPtBins, " #mu#mu pT (GeV)");
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBook2D(ibooker,
             TString::Format("diFilt%dDimuon_recoEtaPt", int(i + 1)),
             dimuonEtaBins,
             "#mu#mu eta",
             dimuonPtBins,
             " #mu#mu pT (GeV)",
             filterNamesLevels[i].first);
  }
  myBook2D(
      ibooker, "diPathDimuon_recoEtaPt", dimuonEtaBins, "#mu#mu eta", dimuonPtBins, " #mu#mu pT (GeV)", triggerPathName);
  // Eta Phi Single
  myBook2D(ibooker, "genMuon_genEtaPhi", muonEtaBins, "#mu eta", muonPhiBins, "#mu phi");
  myBook2D(ibooker, "globMuon_genEtaPhi", muonEtaBins, "#mu eta", muonPhiBins, "#mu phi");
  myBook2D(ibooker, "globMuon_recoEtaPhi", muonEtaBins, "#mu eta", muonPhiBins, "#mu phi");
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBook2D(ibooker,
             TString::Format("filt%dMuon_recoEtaPhi", int(i + 1)),
             muonEtaBins,
             "#mu eta",
             muonPhiBins,
             "#mu phi",
             filterNamesLevels[i].first);
  }
  myBook2D(ibooker, "pathMuon_recoEtaPhi", muonEtaBins, "#mu eta", muonPhiBins, "#mu phi", triggerPathName);
  myBook2D(ibooker, "resultMuon_recoEtaPhi", muonEtaBins, "#mu eta", muonPhiBins, "#mu phi");
  // Rap Pt Double
  myBook2D(ibooker, "genDimuon_genRapPt", dimuonEtaBins, "#mu#mu rapidity", dimuonPtBins, " #mu#mu pT (GeV)");
  myBook2D(ibooker, "globDimuon_genRapPt", dimuonEtaBins, "#mu#mu rapidity", dimuonPtBins, " #mu#mu pT (GeV)");
  myBook2D(ibooker, "globDimuon_recoRapPt", dimuonEtaBins, "#mu#mu rapidity", dimuonPtBins, " #mu#mu pT (GeV)");
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBook2D(ibooker,
             TString::Format("filt%dDimuon_recoRapPt", int(i + 1)),
             dimuonEtaBins,
             "#mu#mu rapidity",
             dimuonPtBins,
             " #mu#mu pT (GeV)",
             filterNamesLevels[i].first);
  }
  myBook2D(ibooker,
           "pathDimuon_recoRapPt",
           dimuonEtaBins,
           "#mu#mu rapidity",
           dimuonPtBins,
           " #mu#mu pT (GeV)",
           triggerPathName);
  myBook2D(ibooker, "resultDimuon_recoRapPt", dimuonEtaBins, "#mu#mu rapidity", dimuonPtBins, " #mu#mu pT (GeV)");
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBook2D(ibooker,
             TString::Format("diFilt%dDimuon_recoRapPt", int(i + 1)),
             dimuonEtaBins,
             "#mu#mu rapidity",
             dimuonPtBins,
             " #mu#mu pT (GeV)",
             filterNamesLevels[i].first);
  }
  myBook2D(ibooker,
           "diPathDimuon_recoRapPt",
           dimuonEtaBins,
           "#mu#mu rapidity",
           dimuonPtBins,
           " #mu#mu pT (GeV)",
           triggerPathName);
  // Pt DR Double
  myBook2D(ibooker, "genDimuon_genPtDR", dimuonPtBins, " #mu#mu pT (GeV)", dimuonDRBins, "#mu#mu #Delta R at IP");
  myBook2D(ibooker, "globDimuon_genPtDR", dimuonPtBins, " #mu#mu pT (GeV)", dimuonDRBins, "#mu#mu #Delta R at IP");
  myBook2D(ibooker, "globDimuon_recoPtDR", dimuonPtBins, " #mu#mu pT (GeV)", dimuonDRBins, "#mu#mu #Delta R at IP");
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBook2D(ibooker,
             TString::Format("filt%dDimuon_recoPtDR", int(i + 1)),
             dimuonPtBins,
             " #mu#mu pT (GeV)",
             dimuonDRBins,
             "#mu#mu #Delta R at IP",
             filterNamesLevels[i].first);
  }
  myBook2D(ibooker,
           "pathDimuon_recoPtDR",
           dimuonPtBins,
           " #mu#mu pT (GeV)",
           dimuonDRBins,
           "#mu#mu #Delta R at IP",
           triggerPathName);
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBook2D(ibooker,
             TString::Format("diFilt%dDimuon_recoPtDR", int(i + 1)),
             dimuonPtBins,
             " #mu#mu pT (GeV)",
             dimuonDRBins,
             "#mu#mu #Delta R at IP",
             filterNamesLevels[i].first);
  }
  myBook2D(ibooker,
           "diPathDimuon_recoPtDR",
           dimuonPtBins,
           " #mu#mu pT (GeV)",
           dimuonDRBins,
           "#mu#mu #Delta R at IP",
           triggerPathName);
  myBook2D(ibooker, "resultDimuon_recoPtDR", dimuonPtBins, " #mu#mu pT (GeV)", dimuonDRBins, "#mu#mu #Delta R at IP");
  // Pt DRpos Double
  myBook2D(ibooker, "globDimuon_recoPtDRpos", dimuonPtBins, " #mu#mu pT (GeV)", dimuonDRBins, "#mu#mu #Delta R in MS");
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBook2D(ibooker,
             TString::Format("filt%dDimuon_recoPtDRpos", int(i + 1)),
             dimuonPtBins,
             " #mu#mu pT (GeV)",
             dimuonDRBins,
             "#mu#mu #Delta R in MS",
             filterNamesLevels[i].first);
  }
  myBook2D(ibooker,
           "pathDimuon_recoPtDRpos",
           dimuonPtBins,
           " #mu#mu pT (GeV)",
           dimuonDRBins,
           "#mu#mu #Delta R in MS",
           triggerPathName);
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBook2D(ibooker,
             TString::Format("diFilt%dDimuon_recoPtDRpos", int(i + 1)),
             dimuonPtBins,
             " #mu#mu pT (GeV)",
             dimuonDRBins,
             "#mu#mu #Delta R in MS",
             filterNamesLevels[i].first);
  }
  myBook2D(ibooker,
           "diPathDimuon_recoPtDRpos",
           dimuonPtBins,
           " #mu#mu pT (GeV)",
           dimuonDRBins,
           "#mu#mu #Delta R in MS",
           triggerPathName);
  myBook2D(
      ibooker, "resultDimuon_recoPtDRpos", dimuonPtBins, " #mu#mu pT (GeV)", dimuonDRBins, "#mu#mu #Delta R in MS");

  // Matching
  myBook2D(ibooker, "globGen_deltaEtaDeltaPhi", deltaEtaBins, "#Delta eta", deltaPhiBins, "#Delta phi");
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBook2D(ibooker,
             TString::Format("filt%dGlob_deltaEtaDeltaPhi", int(i + 1)),
             deltaEtaBins,
             "#Delta eta",
             deltaPhiBins,
             "#Delta phi",
             filterNamesLevels[i].first);
  }
  myBook2D(
      ibooker, "pathGlob_deltaEtaDeltaPhi", deltaEtaBins, "#Delta eta", deltaPhiBins, "#Delta phi", triggerPathName);
  // Size of containers
  vector<double> sizeBins;
  sizeBins.push_back(10);
  sizeBins.push_back(0);
  sizeBins.push_back(10);
  myBook1D(ibooker, "genMuon_size", sizeBins, "container size");
  myBook1D(ibooker, "globMuon_size", sizeBins, "container size");
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    myBook1D(
        ibooker, TString::Format("filt%dMuon_size", int(i + 1)), sizeBins, "container size", filterNamesLevels[i].first);
  }
  myBook1D(ibooker, "pathMuon_size", sizeBins, "container size", triggerPathName);
}

void HeavyFlavorValidation::analyze(const Event &iEvent, const EventSetup &iSetup) {
  if (filterNamesLevels.empty()) {
    return;
  }
  //access the containers and create LeafCandidate copies
  vector<LeafCandidate> genMuons;
  Handle<GenParticleCollection> genParticles;
  iEvent.getByToken(genParticlesToken, genParticles);
  if (genParticles.isValid()) {
    for (GenParticleCollection::const_iterator p = genParticles->begin(); p != genParticles->end(); ++p) {
      if (p->status() == 1 && std::abs(p->pdgId()) == 13 &&
          (find(motherIDs.begin(), motherIDs.end(), -1) != motherIDs.end() ||
           find(motherIDs.begin(), motherIDs.end(), getMotherId(&(*p))) != motherIDs.end())) {
        genMuons.push_back(*p);
      }
    }
  } else {
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Could not access GenParticleCollection" << endl;
  }
  sort(genMuons.begin(), genMuons.end(), GreaterByPt<LeafCandidate>());
  ME["genMuon_size"]->Fill(genMuons.size());
  LogDebug("HLTriggerOfflineHeavyFlavor")
      << "GenParticleCollection from " << genParticlesTag << " has size: " << genMuons.size() << endl;

  vector<LeafCandidate> globMuons;
  vector<LeafCandidate> globMuons_position;
  Handle<MuonCollection> recoMuonsHandle;
  iEvent.getByToken(recoMuonsToken, recoMuonsHandle);
  if (recoMuonsHandle.isValid()) {
    for (MuonCollection::const_iterator p = recoMuonsHandle->begin(); p != recoMuonsHandle->end(); ++p) {
      if (p->isGlobalMuon()) {
        globMuons.push_back(*p);
        globMuons_position.push_back(LeafCandidate(p->charge(),
                                                   math::XYZTLorentzVector(p->outerTrack()->innerPosition().x(),
                                                                           p->outerTrack()->innerPosition().y(),
                                                                           p->outerTrack()->innerPosition().z(),
                                                                           0.)));
      }
    }
  } else {
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Could not access reco Muons" << endl;
  }
  ME["globMuon_size"]->Fill(globMuons.size());
  LogDebug("HLTriggerOfflineHeavyFlavor")
      << "Global Muons from " << recoMuonsTag << " has size: " << globMuons.size() << endl;

  // access RAW trigger event
  vector<vector<LeafCandidate> > muonsAtFilter;
  vector<vector<LeafCandidate> > muonPositionsAtFilter;
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    muonsAtFilter.push_back(vector<LeafCandidate>());
    muonPositionsAtFilter.push_back(vector<LeafCandidate>());
  }
  Handle<TriggerEventWithRefs> rawTriggerEvent;
  iEvent.getByToken(triggerSummaryRAWTag, rawTriggerEvent);
  if (rawTriggerEvent.isValid()) {
    for (size_t i = 0; i < filterNamesLevels.size(); i++) {
      size_t index = rawTriggerEvent->filterIndex(InputTag(filterNamesLevels[i].first, "", triggerProcessName));
      if (index < rawTriggerEvent->size()) {
        if (filterNamesLevels[i].second == 1) {
          vector<L1MuonParticleRef> l1Cands;
          rawTriggerEvent->getObjects(index, TriggerL1Mu, l1Cands);
          for (size_t j = 0; j < l1Cands.size(); j++) {
            muonsAtFilter[i].push_back(*l1Cands[j]);
          }
        } else {
          vector<RecoChargedCandidateRef> hltCands;
          rawTriggerEvent->getObjects(index, TriggerMuon, hltCands);
          for (size_t j = 0; j < hltCands.size(); j++) {
            muonsAtFilter[i].push_back(*hltCands[j]);
            if (filterNamesLevels[i].second == 2) {
              muonPositionsAtFilter[i].push_back(
                  LeafCandidate(hltCands[j]->charge(),
                                math::XYZTLorentzVector(hltCands[j]->track()->innerPosition().x(),
                                                        hltCands[j]->track()->innerPosition().y(),
                                                        hltCands[j]->track()->innerPosition().z(),
                                                        0.)));
            }
          }
        }
      }
      ME[TString::Format("filt%dMuon_size", int(i + 1))]->Fill(muonsAtFilter[i].size());
      LogDebug("HLTriggerOfflineHeavyFlavor")
          << "Filter \"" << filterNamesLevels[i].first << "\" has " << muonsAtFilter[i].size() << " muons" << endl;
    }
  } else {
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Could not access RAWTriggerEvent" << endl;
  }

  // access AOD trigger event
  vector<LeafCandidate> pathMuons;
  Handle<TriggerEvent> aodTriggerEvent;
  iEvent.getByToken(triggerSummaryAODTag, aodTriggerEvent);
  if (aodTriggerEvent.isValid()) {
    TriggerObjectCollection allObjects = aodTriggerEvent->getObjects();
    for (int i = 0; i < aodTriggerEvent->sizeFilters(); i++) {
      if (aodTriggerEvent->filterTag(i) == InputTag((filterNamesLevels.end() - 1)->first, "", triggerProcessName)) {
        Keys keys = aodTriggerEvent->filterKeys(i);
        for (size_t j = 0; j < keys.size(); j++) {
          pathMuons.push_back(LeafCandidate(
              allObjects[keys[j]].id() > 0 ? 1 : -1,
              math::PtEtaPhiMLorentzVector(
                  allObjects[keys[j]].pt(), allObjects[keys[j]].eta(), allObjects[keys[j]].phi(), muonMass)));
        }
      }
    }
    ME["pathMuon_size"]->Fill(pathMuons.size());
    LogDebug("HLTriggerOfflineHeavyFlavor")
        << "Path \"" << triggerPathName << "\" has " << pathMuons.size() << " muons at last filter \""
        << (filterNamesLevels.end() - 1)->first << "\"" << endl;
  } else {
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Could not access AODTriggerEvent" << endl;
  }

  // access Trigger Results
  bool triggerFired = false;
  Handle<TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken, triggerResults);
  if (triggerResults.isValid()) {
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Successfully initialized " << triggerResultsTag << endl;
    const edm::TriggerNames &triggerNames = iEvent.triggerNames(*triggerResults);
    bool hlt_exists = false;
    for (unsigned int i = 0; i != triggerNames.size(); i++) {
      TString hlt_name = triggerNames.triggerName(i);
      if (hlt_name.Contains(triggerPathName)) {
        triggerFired = triggerResults->accept(i);
        hlt_exists = true;
        break;
      }
    }
    if (!hlt_exists) {
      LogDebug("HLTriggerOfflineHeavyFlavor") << triggerResultsTag << " has no trigger: " << triggerPathName << endl;
    }
  } else {
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Could not initialize " << triggerResultsTag << endl;
  }

  //create matching maps
  vector<int> glob_gen(genMuons.size(), -1);
  match(ME["globGen_deltaEtaDeltaPhi"], genMuons, globMuons, genGlobDeltaRMatchingCut, glob_gen);
  vector<vector<int> > filt_glob;
  for (size_t i = 0; i < filterNamesLevels.size(); i++) {
    filt_glob.push_back(vector<int>(globMuons.size(), -1));
    if (filterNamesLevels[i].second == 1) {
      match(ME[TString::Format("filt%dGlob_deltaEtaDeltaPhi", int(i + 1))],
            globMuons_position,
            muonsAtFilter[i],
            globL1DeltaRMatchingCut,
            filt_glob[i]);
    } else if (filterNamesLevels[i].second == 2) {
      match(ME[TString::Format("filt%dGlob_deltaEtaDeltaPhi", int(i + 1))],
            globMuons_position,
            muonPositionsAtFilter[i],
            globL2DeltaRMatchingCut,
            filt_glob[i]);
    } else if (filterNamesLevels[i].second > 2) {
      match(ME[TString::Format("filt%dGlob_deltaEtaDeltaPhi", int(i + 1))],
            globMuons,
            muonsAtFilter[i],
            globL3DeltaRMatchingCut,
            filt_glob[i]);
    }
  }
  vector<int> path_glob(globMuons.size(), -1);
  if ((filterNamesLevels.end() - 1)->second == 1) {
    match(ME["pathGlob_deltaEtaDeltaPhi"], globMuons_position, pathMuons, globL1DeltaRMatchingCut, path_glob);
  } else if ((filterNamesLevels.end() - 1)->second == 2) {
    match(ME["pathGlob_deltaEtaDeltaPhi"], globMuons, pathMuons, globL2DeltaRMatchingCut, path_glob);
  } else if ((filterNamesLevels.end() - 1)->second > 2) {
    match(ME["pathGlob_deltaEtaDeltaPhi"], globMuons, pathMuons, globL3DeltaRMatchingCut, path_glob);
  }

  //fill histos
  bool first = true;
  for (size_t i = 0; i < genMuons.size(); i++) {
    ME["genMuon_genEtaPt"]->Fill(genMuons[i].eta(), genMuons[i].pt());
    ME["genMuon_genEtaPhi"]->Fill(genMuons[i].eta(), genMuons[i].phi());
    if (glob_gen[i] != -1) {
      ME["resGlobGen_genEtaPt"]->Fill(
          genMuons[i].eta(), genMuons[i].pt(), (globMuons[glob_gen[i]].pt() - genMuons[i].pt()) / genMuons[i].pt());
      ME["globMuon_genEtaPt"]->Fill(genMuons[i].eta(), genMuons[i].pt());
      ME["globMuon_genEtaPhi"]->Fill(genMuons[i].eta(), genMuons[i].phi());
      ME["globMuon_recoEtaPt"]->Fill(globMuons[glob_gen[i]].eta(), globMuons[glob_gen[i]].pt());
      ME["globMuon_recoEtaPhi"]->Fill(globMuons[glob_gen[i]].eta(), globMuons[glob_gen[i]].phi());
      for (size_t f = 0; f < filterNamesLevels.size(); f++) {
        if (filt_glob[f][glob_gen[i]] != -1) {
          ME[TString::Format("resFilt%dGlob_recoEtaPt", int(f + 1))]->Fill(
              globMuons[glob_gen[i]].eta(),
              globMuons[glob_gen[i]].pt(),
              (muonsAtFilter[f][filt_glob[f][glob_gen[i]]].pt() - globMuons[glob_gen[i]].pt()) /
                  globMuons[glob_gen[i]].pt());
          ME[TString::Format("filt%dMuon_recoEtaPt", int(f + 1))]->Fill(globMuons[glob_gen[i]].eta(),
                                                                        globMuons[glob_gen[i]].pt());
          ME[TString::Format("filt%dMuon_recoEtaPhi", int(f + 1))]->Fill(globMuons[glob_gen[i]].eta(),
                                                                         globMuons[glob_gen[i]].phi());
        } else {
          break;
        }
      }
      if (path_glob[glob_gen[i]] != -1) {
        ME["resPathGlob_recoEtaPt"]->Fill(
            globMuons[glob_gen[i]].eta(),
            globMuons[glob_gen[i]].pt(),
            (pathMuons[path_glob[glob_gen[i]]].pt() - globMuons[glob_gen[i]].pt()) / globMuons[glob_gen[i]].pt());
        ME["pathMuon_recoEtaPt"]->Fill(globMuons[glob_gen[i]].eta(), globMuons[glob_gen[i]].pt());
        ME["pathMuon_recoEtaPhi"]->Fill(globMuons[glob_gen[i]].eta(), globMuons[glob_gen[i]].phi());
      }
      //highest pt muon
      if (first) {
        first = false;
        if (triggerFired) {
          ME["resultMuon_recoEtaPt"]->Fill(globMuons[glob_gen[i]].eta(), globMuons[glob_gen[i]].pt());
          ME["resultMuon_recoEtaPhi"]->Fill(globMuons[glob_gen[i]].eta(), globMuons[glob_gen[i]].phi());
        }
      }
    }
  }

  //fill dimuon histograms (highest pT, opposite charge)
  int secondMuon = 0;
  for (size_t j = 1; j < genMuons.size(); j++) {
    if (genMuons[0].charge() * genMuons[j].charge() == -1) {
      secondMuon = j;
      break;
    }
  }
  if (secondMuon > 0) {
    //two generated
    double genDimuonPt = (genMuons[0].p4() + genMuons[secondMuon].p4()).pt();
    double genDimuonEta = (genMuons[0].p4() + genMuons[secondMuon].p4()).eta();
    double genDimuonRap = (genMuons[0].p4() + genMuons[secondMuon].p4()).Rapidity();
    double genDimuonDR = deltaR<LeafCandidate, LeafCandidate>(genMuons[0], genMuons[secondMuon]);
    bool highPt = genMuons[0].pt() > 7. && genMuons[secondMuon].pt() > 7;
    ME["genDimuon_genEtaPt"]->Fill(genDimuonEta, genDimuonPt);
    ME["genDimuon_genRapPt"]->Fill(genDimuonRap, genDimuonPt);
    if (highPt)
      ME["genDimuon_genPtDR"]->Fill(genDimuonPt, genDimuonDR);
    //two global
    if (glob_gen[0] != -1 && glob_gen[secondMuon] != -1) {
      ME["globDimuon_genEtaPt"]->Fill(genDimuonEta, genDimuonPt);
      ME["globDimuon_genRapPt"]->Fill(genDimuonRap, genDimuonPt);
      if (highPt)
        ME["globDimuon_genPtDR"]->Fill(genDimuonPt, genDimuonDR);
      double globDimuonPt = (globMuons[glob_gen[0]].p4() + globMuons[glob_gen[secondMuon]].p4()).pt();
      double globDimuonEta = (globMuons[glob_gen[0]].p4() + globMuons[glob_gen[secondMuon]].p4()).eta();
      double globDimuonRap = (globMuons[glob_gen[0]].p4() + globMuons[glob_gen[secondMuon]].p4()).Rapidity();
      double globDimuonDR =
          deltaR<LeafCandidate, LeafCandidate>(globMuons[glob_gen[0]], globMuons[glob_gen[secondMuon]]);
      double globDimuonDRpos = deltaR<LeafCandidate, LeafCandidate>(globMuons_position[glob_gen[0]],
                                                                    globMuons_position[glob_gen[secondMuon]]);
      ME["globDimuon_recoEtaPt"]->Fill(globDimuonEta, globDimuonPt);
      ME["globDimuon_recoRapPt"]->Fill(globDimuonRap, globDimuonPt);
      if (highPt)
        ME["globDimuon_recoPtDR"]->Fill(globDimuonPt, globDimuonDR);
      if (highPt)
        ME["globDimuon_recoPtDRpos"]->Fill(globDimuonPt, globDimuonDRpos);
      //two filter objects
      for (size_t f = 0; f < filterNamesLevels.size(); f++) {
        if (filt_glob[f][glob_gen[0]] != -1 && filt_glob[f][glob_gen[secondMuon]] != -1) {
          ME[TString::Format("diFilt%dDimuon_recoEtaPt", int(f + 1))]->Fill(globDimuonEta, globDimuonPt);
          ME[TString::Format("diFilt%dDimuon_recoRapPt", int(f + 1))]->Fill(globDimuonRap, globDimuonPt);
          if (highPt)
            ME[TString::Format("diFilt%dDimuon_recoPtDR", int(f + 1))]->Fill(globDimuonPt, globDimuonDR);
          if (highPt)
            ME[TString::Format("diFilt%dDimuon_recoPtDRpos", int(f + 1))]->Fill(globDimuonPt, globDimuonDRpos);
        } else {
          break;
        }
      }
      //one filter object
      for (size_t f = 0; f < filterNamesLevels.size(); f++) {
        if (filt_glob[f][glob_gen[0]] != -1 || filt_glob[f][glob_gen[secondMuon]] != -1) {
          ME[TString::Format("filt%dDimuon_recoEtaPt", int(f + 1))]->Fill(globDimuonEta, globDimuonPt);
          ME[TString::Format("filt%dDimuon_recoRapPt", int(f + 1))]->Fill(globDimuonRap, globDimuonPt);
          if (highPt)
            ME[TString::Format("filt%dDimuon_recoPtDR", int(f + 1))]->Fill(globDimuonPt, globDimuonDR);
          if (highPt)
            ME[TString::Format("filt%dDimuon_recoPtDRpos", int(f + 1))]->Fill(globDimuonPt, globDimuonDRpos);
        } else {
          break;
        }
      }
      //two path objects
      if (path_glob[glob_gen[0]] != -1 && path_glob[glob_gen[secondMuon]] != -1) {
        ME["diPathDimuon_recoEtaPt"]->Fill(globDimuonEta, globDimuonPt);
        ME["diPathDimuon_recoRapPt"]->Fill(globDimuonRap, globDimuonPt);
        if (highPt)
          ME["diPathDimuon_recoPtDR"]->Fill(globDimuonPt, globDimuonDR);
        if (highPt)
          ME["diPathDimuon_recoPtDRpos"]->Fill(globDimuonPt, globDimuonDRpos);
      }
      //one path object
      if (path_glob[glob_gen[0]] != -1 || path_glob[glob_gen[secondMuon]] != -1) {
        ME["pathDimuon_recoEtaPt"]->Fill(globDimuonEta, globDimuonPt);
        ME["pathDimuon_recoRapPt"]->Fill(globDimuonRap, globDimuonPt);
        if (highPt)
          ME["pathDimuon_recoPtDR"]->Fill(globDimuonPt, globDimuonDR);
        if (highPt)
          ME["pathDimuon_recoPtDRpos"]->Fill(globDimuonPt, globDimuonDRpos);
      }
      //trigger result
      if (triggerFired) {
        ME["resultDimuon_recoEtaPt"]->Fill(globDimuonEta, globDimuonPt);
        ME["resultDimuon_recoRapPt"]->Fill(globDimuonRap, globDimuonPt);
        if (highPt)
          ME["resultDimuon_recoPtDR"]->Fill(globDimuonPt, globDimuonDR);
        if (highPt)
          ME["resultDimuon_recoPtDRpos"]->Fill(globDimuonPt, globDimuonDRpos);
      }
    }
  }
}

int HeavyFlavorValidation::getMotherId(const Candidate *p) {
  const Candidate *mother = p->mother();
  if (mother) {
    if (mother->pdgId() == p->pdgId()) {
      return getMotherId(mother);
    } else {
      return mother->pdgId();
    }
  } else {
    return 0;
  }
}

void HeavyFlavorValidation::match(MonitorElement *me,
                                  vector<LeafCandidate> &from,
                                  vector<LeafCandidate> &to,
                                  double dRMatchingCut,
                                  vector<int> &map) {
  vector<double> dR(from.size());
  for (size_t i = 0; i < from.size(); i++) {
    map[i] = -1;
    dR[i] = 10.;
    //find closest
    for (size_t j = 0; j < to.size(); j++) {
      double dRtmp = deltaR<double>(from[i].eta(), from[i].phi(), to[j].eta(), to[j].phi());
      if (dRtmp < dR[i]) {
        dR[i] = dRtmp;
        map[i] = j;
      }
    }
    //fill matching histo
    if (map[i] != -1) {
      me->Fill(to[map[i]].eta() - from[i].eta(), deltaPhi<double>(to[map[i]].phi(), from[i].phi()));
    }
    //apply matching cut
    if (dR[i] > dRMatchingCut) {
      map[i] = -1;
    }
    //remove duplication
    if (map[i] != -1) {
      for (size_t k = 0; k < i; k++) {
        if (map[k] != -1 && map[i] == map[k]) {
          if (dR[i] < dR[k]) {
            map[k] = -1;
          } else {
            map[i] = -1;
          }
          break;
        }
      }
    }
  }
}

void HeavyFlavorValidation::myBook2D(DQMStore::IBooker &ibooker,
                                     TString name,
                                     vector<double> &ptBins,
                                     TString ptLabel,
                                     vector<double> &etaBins,
                                     TString etaLabel,
                                     TString title) {
  //   dqmStore->setCurrentFolder(dqmFolder+"/"+folder);
  int ptN = ptBins.size() == 3 ? (int)ptBins[0] + 1 : ptBins.size();
  Double_t *pt = new Double_t[ptN];
  for (int i = 0; i < ptN; i++) {
    pt[i] = ptBins.size() == 3 ? ptBins[1] + i * (ptBins[2] - ptBins[1]) / ptBins[0] : ptBins[i];
  }
  int etaN = etaBins.size() == 3 ? (int)etaBins[0] + 1 : etaBins.size();
  Double_t *eta = new Double_t[etaN];
  for (int i = 0; i < etaN; i++) {
    eta[i] = etaBins.size() == 3 ? etaBins[1] + i * (etaBins[2] - etaBins[1]) / etaBins[0] : etaBins[i];
  }
  TH2F *h = new TH2F(name, name, ptN - 1, pt, etaN - 1, eta);
  h->SetXTitle(ptLabel);
  h->SetYTitle(etaLabel);
  h->SetTitle(title);
  ME[name] = ibooker.book2D(name.Data(), h);
  delete h;
}

void HeavyFlavorValidation::myBookProfile2D(DQMStore::IBooker &ibooker,
                                            TString name,
                                            vector<double> &ptBins,
                                            TString ptLabel,
                                            vector<double> &etaBins,
                                            TString etaLabel,
                                            TString title) {
  //   dqmStore->setCurrentFolder(dqmFolder+"/"+folder);
  int ptN = ptBins.size() == 3 ? (int)ptBins[0] + 1 : ptBins.size();
  Double_t *pt = new Double_t[ptN];
  for (int i = 0; i < ptN; i++) {
    pt[i] = ptBins.size() == 3 ? ptBins[1] + i * (ptBins[2] - ptBins[1]) / ptBins[0] : ptBins[i];
  }
  int etaN = etaBins.size() == 3 ? (int)etaBins[0] + 1 : etaBins.size();
  Double_t *eta = new Double_t[etaN];
  for (int i = 0; i < etaN; i++) {
    eta[i] = etaBins.size() == 3 ? etaBins[1] + i * (etaBins[2] - etaBins[1]) / etaBins[0] : etaBins[i];
  }
  TProfile2D *h = new TProfile2D(name, name, ptN - 1, pt, etaN - 1, eta);
  h->SetXTitle(ptLabel);
  h->SetYTitle(etaLabel);
  h->SetTitle(title);
  ME[name] = ibooker.bookProfile2D(name.Data(), h);
  delete h;
}

void HeavyFlavorValidation::myBook1D(
    DQMStore::IBooker &ibooker, TString name, vector<double> &bins, TString label, TString title) {
  //   dqmStore->setCurrentFolder(dqmFolder+"/"+folder);
  int binsN = bins.size() == 3 ? (int)bins[0] + 1 : bins.size();
  Double_t *myBins = new Double_t[binsN];
  for (int i = 0; i < binsN; i++) {
    myBins[i] = bins.size() == 3 ? bins[1] + i * (bins[2] - bins[1]) / bins[0] : bins[i];
  }
  TH1F *h = new TH1F(name, name, binsN - 1, myBins);
  h->SetXTitle(label);
  h->SetTitle(title);
  ME[name] = ibooker.book1D(name.Data(), h);
  delete h;
}

int HeavyFlavorValidation::getFilterLevel(const std::string &moduleName, const HLTConfigProvider &hltConfig) {
  // helper lambda to check if a string contains a substring
  const auto contains = [](const std::string &s, const std::string &sub) -> bool {
    return s.find(sub) != std::string::npos;
  };

  // helper lambda to check if a string contains any of a list of substrings
  const auto containsAny = [](const std::string &s, const std::vector<std::string> &subs) -> bool {
    for (const auto &sub : subs) {
      if (s.find(sub) != std::string::npos)
        return true;
    }
    return false;
  };

  // helper lambda to check if string s is any of the strings in vector ms
  const auto isAnyOf = [](const std::string &s, const std::vector<std::string> &ms) -> bool {
    for (const auto &m : ms) {
      if (s == m)
        return true;
    }
    return false;
  };

  // tmadlener, 20.08.2017:
  // define the valid module names for the different "levels", to add a little bit more stability
  // to the checking compared to just doing some name matching.
  // Note, that the name matching is not completely remved, since at level 4 and 5 some of the
  // valid modules are the same, so that the name matching is still needed.
  // With the current definition this yields the exact same levels as before, but weeds out some
  // of the "false" positives at level 3 (naming matches also to some HLTMuonL1TFilter modules due to
  // the 'forIterL3' in the name)
  const std::string l1Filter = "HLTMuonL1TFilter";
  const std::string l2Filter = "HLTMuonL2FromL1TPreFilter";
  const std::vector<std::string> l3Filters = {"HLTMuonDimuonL3Filter", "HLTMuonL3PreFilter"};
  const std::vector<std::string> l4Filters = {
      "HLTDisplacedmumuFilter", "HLTDiMuonGlbTrkFilter", "HLTMuonTrackMassFilter"};
  const std::vector<std::string> l5Filters = {"HLTmumutkFilter", "HLT2MuonMuonDZ", "HLTDisplacedmumuFilter"};

  if (contains(moduleName, "Filter") && hltConfig.moduleEDMType(moduleName) == "EDFilter") {
    if (contains(moduleName, "L1") && !contains(moduleName, "ForIterL3") &&
        hltConfig.moduleType(moduleName) == l1Filter) {
      return 1;
    }
    if (contains(moduleName, "L2") && hltConfig.moduleType(moduleName) == l2Filter) {
      return 2;
    }
    if (contains(moduleName, "L3") && isAnyOf(hltConfig.moduleType(moduleName), l3Filters)) {
      return 3;
    }
    if (containsAny(moduleName, {"DisplacedmumuFilter", "DiMuon", "MuonL3Filtered", "TrackMassFiltered"}) &&
        isAnyOf(hltConfig.moduleType(moduleName), l4Filters)) {
      return 4;
    }
    if (containsAny(moduleName, {"Vertex", "Dz"}) && isAnyOf(hltConfig.moduleType(moduleName), l5Filters)) {
      return 5;
    }
  }

  return -1;
}

HeavyFlavorValidation::~HeavyFlavorValidation() {}

//define this as a plug-in
DEFINE_FWK_MODULE(HeavyFlavorValidation);
