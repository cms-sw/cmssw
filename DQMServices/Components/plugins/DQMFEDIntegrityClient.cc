/*
 * \file DQMFEDIntegrityClient.cc
 * \author M. Marienfeld
 * Last Update:
 *
 * Description: Summing up FED entries from all subdetectors.
 *
 */

#include <string>
#include <vector>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//

class DQMFEDIntegrityClient : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;
  DQMFEDIntegrityClient(const edm::ParameterSet&);
  ~DQMFEDIntegrityClient() override = default;

protected:
  void beginJob() override;
  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  void beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c) override;
  void endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c) override;

  void endRun(const edm::Run& r, const edm::EventSetup& c) override;
  void endJob() override;

private:
  void initialize();
  void fillHistograms();

  edm::ParameterSet parameters_;

  DQMStore* dbe_;

  // ---------- member data ----------

  int NBINS;
  float XMIN, XMAX;
  float SummaryContent[10];

  MonitorElement* FedEntries;
  MonitorElement* FedFatal;
  MonitorElement* FedNonFatal;

  MonitorElement* reportSummary;
  MonitorElement* reportSummaryContent[10];
  MonitorElement* reportSummaryMap;

  bool fillInEventloop;
  bool fillOnEndRun;
  bool fillOnEndJob;
  bool fillOnEndLumi;
  std::string moduleName;
  std::string fedFolderName;
};

// -----------------------------
//  constructors and destructor
// -----------------------------

DQMFEDIntegrityClient::DQMFEDIntegrityClient(const edm::ParameterSet& ps) {
  parameters_ = ps;
  initialize();
  fillInEventloop = ps.getUntrackedParameter<bool>("fillInEventloop", false);
  fillOnEndRun = ps.getUntrackedParameter<bool>("fillOnEndRun", false);
  fillOnEndJob = ps.getUntrackedParameter<bool>("fillOnEndJob", false);
  fillOnEndLumi = ps.getUntrackedParameter<bool>("fillOnEndLumi", true);
  moduleName = ps.getUntrackedParameter<std::string>("moduleName", "FED");
  fedFolderName = ps.getUntrackedParameter<std::string>("fedFolderName", "FEDIntegrity");
}

void DQMFEDIntegrityClient::initialize() {
  // get back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
}

void DQMFEDIntegrityClient::beginJob() {
  NBINS = 850;
  XMIN = 0.;
  XMAX = 850.;

  dbe_ = edm::Service<DQMStore>().operator->();

  // ----------------------------------------------------------------------------------
  std::string currentFolder = moduleName + "/" + fedFolderName;
  dbe_->setCurrentFolder(currentFolder);

  FedEntries = dbe_->book1D("FedEntries", "FED Entries", NBINS, XMIN, XMAX);
  FedFatal = dbe_->book1D("FedFatal", "FED Fatal Errors", NBINS, XMIN, XMAX);
  FedNonFatal = dbe_->book1D("FedNonFatal", "FED Non Fatal Errors", NBINS, XMIN, XMAX);

  FedEntries->setAxisTitle("", 1);
  FedFatal->setAxisTitle("", 1);
  FedNonFatal->setAxisTitle("", 1);

  FedEntries->setAxisTitle("", 2);
  FedFatal->setAxisTitle("", 2);
  FedNonFatal->setAxisTitle("", 2);

  FedEntries->setBinLabel(11, "PIXEL", 1);
  FedEntries->setBinLabel(221, "SIST", 1);
  FedEntries->setBinLabel(606, "EE", 1);
  FedEntries->setBinLabel(628, "EB", 1);
  FedEntries->setBinLabel(651, "EE", 1);
  FedEntries->setBinLabel(550, "ES", 1);
  FedEntries->setBinLabel(716, "HCAL", 1);
  FedEntries->setBinLabel(754, "CSC", 1);
  FedEntries->setBinLabel(772, "DT", 1);
  FedEntries->setBinLabel(791, "RPC", 1);
  FedEntries->setBinLabel(804, "L1T", 1);

  FedFatal->setBinLabel(11, "PIXEL", 1);
  FedFatal->setBinLabel(221, "SIST", 1);
  FedFatal->setBinLabel(606, "EE", 1);
  FedFatal->setBinLabel(628, "EB", 1);
  FedFatal->setBinLabel(651, "EE", 1);
  FedFatal->setBinLabel(550, "ES", 1);
  FedFatal->setBinLabel(716, "HCAL", 1);
  FedFatal->setBinLabel(754, "CSC", 1);
  FedFatal->setBinLabel(772, "DT", 1);
  FedFatal->setBinLabel(791, "RPC", 1);
  FedFatal->setBinLabel(804, "L1T", 1);

  FedNonFatal->setBinLabel(11, "PIXEL", 1);
  FedNonFatal->setBinLabel(221, "SIST", 1);
  FedNonFatal->setBinLabel(606, "EE", 1);
  FedNonFatal->setBinLabel(628, "EB", 1);
  FedNonFatal->setBinLabel(651, "EE", 1);
  FedNonFatal->setBinLabel(550, "ES", 1);
  FedNonFatal->setBinLabel(716, "HCAL", 1);
  FedNonFatal->setBinLabel(754, "CSC", 1);
  FedNonFatal->setBinLabel(772, "DT", 1);
  FedNonFatal->setBinLabel(791, "RPC", 1);
  FedNonFatal->setBinLabel(804, "L1T", 1);

  //-----------------------------------------------------------------------------------
  currentFolder = moduleName + "/EventInfo";
  dbe_->setCurrentFolder(currentFolder);

  reportSummary = dbe_->bookFloat("reportSummary");

  int nSubsystems = 10;

  if (reportSummary)
    reportSummary->Fill(1.);

  currentFolder = moduleName + "/EventInfo/reportSummaryContents";
  dbe_->setCurrentFolder(currentFolder);

  reportSummaryContent[0] = dbe_->bookFloat("CSC FEDs");
  reportSummaryContent[1] = dbe_->bookFloat("DT FEDs");
  reportSummaryContent[2] = dbe_->bookFloat("EB FEDs");
  reportSummaryContent[3] = dbe_->bookFloat("EE FEDs");
  reportSummaryContent[4] = dbe_->bookFloat("ES FEDs");
  reportSummaryContent[5] = dbe_->bookFloat("Hcal FEDs");
  reportSummaryContent[6] = dbe_->bookFloat("L1T FEDs");
  reportSummaryContent[7] = dbe_->bookFloat("Pixel FEDs");
  reportSummaryContent[8] = dbe_->bookFloat("RPC FEDs");
  reportSummaryContent[9] = dbe_->bookFloat("SiStrip FEDs");

  // initialize reportSummaryContents to 1
  for (int i = 0; i < nSubsystems; ++i) {
    SummaryContent[i] = 1.;
    reportSummaryContent[i]->Fill(1.);
  }

  currentFolder = moduleName + "/EventInfo";
  dbe_->setCurrentFolder(currentFolder);

  reportSummaryMap = dbe_->book2D("reportSummaryMap", "FED Report Summary Map", 1, 1, 2, 10, 1, 11);

  reportSummaryMap->setAxisTitle("", 1);
  reportSummaryMap->setAxisTitle("", 2);

  reportSummaryMap->setBinLabel(1, " ", 1);
  reportSummaryMap->setBinLabel(10, "CSC", 2);
  reportSummaryMap->setBinLabel(9, "DT", 2);
  reportSummaryMap->setBinLabel(8, "EB", 2);
  reportSummaryMap->setBinLabel(7, "EE", 2);
  reportSummaryMap->setBinLabel(6, "ES", 2);
  reportSummaryMap->setBinLabel(5, "Hcal", 2);
  reportSummaryMap->setBinLabel(4, "L1T", 2);
  reportSummaryMap->setBinLabel(3, "Pixel", 2);
  reportSummaryMap->setBinLabel(2, "RPC", 2);
  reportSummaryMap->setBinLabel(1, "SiStrip", 2);
}

void DQMFEDIntegrityClient::beginRun(const edm::Run& r, const edm::EventSetup& context) {}

void DQMFEDIntegrityClient::analyze(const edm::Event& e, const edm::EventSetup& context) {
  if (fillInEventloop)
    fillHistograms();
}

void DQMFEDIntegrityClient::beginLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c) {}

void DQMFEDIntegrityClient::endLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& context) {
  if (fillOnEndLumi)
    fillHistograms();
}

void DQMFEDIntegrityClient::fillHistograms() {
  // FED Entries

  std::vector<std::string> entries;
  entries.push_back("CSC/" + fedFolderName + "/FEDEntries");
  entries.push_back("DT/" + fedFolderName + "/FEDEntries");
  entries.push_back("EcalBarrel/" + fedFolderName + "/FEDEntries");
  entries.push_back("EcalEndcap/" + fedFolderName + "/FEDEntries");
  entries.push_back("EcalPreshower/" + fedFolderName + "/FEDEntries");
  entries.push_back("Hcal/" + fedFolderName + "/FEDEntries");
  entries.push_back("L1T/" + fedFolderName + "/FEDEntries");
  entries.push_back("Pixel/" + fedFolderName + "/FEDEntries");
  entries.push_back("RPC/" + fedFolderName + "/FEDEntries");
  entries.push_back("SiStrip/" + fedFolderName + "/FEDEntries");

  for (auto ent = entries.begin(); ent != entries.end(); ++ent) {
    if (!(dbe_->get(*ent))) {
      continue;
    }

    MonitorElement* me = dbe_->get(*ent);
    if (TH1F* rootHisto = me->getTH1F()) {
      int Nbins = me->getNbinsX();
      float entry = 0.;
      int xmin = (int)rootHisto->GetXaxis()->GetXmin();
      if (*ent == "L1T/" + fedFolderName + "/FEDEntries")
        xmin = xmin + 800;
      if (*ent == "DT/" + fedFolderName + "/FEDEntries")
        xmin = 770;  //Real DT FEDIDs are 1369-1371

      for (int bin = 1; bin <= Nbins; ++bin) {
        int id = xmin + bin;
        entry = rootHisto->GetBinContent(bin);
        if (entry > 0.)
          FedEntries->setBinContent(id, entry);
      }
    }
  }

  // FED Fatal

  int nSubsystems = 10;

  std::vector<std::string> fatal;
  fatal.push_back("CSC/" + fedFolderName + "/FEDFatal");
  fatal.push_back("DT/" + fedFolderName + "/FEDFatal");
  fatal.push_back("EcalBarrel/" + fedFolderName + "/FEDFatal");
  fatal.push_back("EcalEndcap/" + fedFolderName + "/FEDFatal");
  fatal.push_back("EcalPreshower/" + fedFolderName + "/FEDFatal");
  fatal.push_back("Hcal/" + fedFolderName + "/FEDFatal");
  fatal.push_back("L1T/" + fedFolderName + "/FEDFatal");
  fatal.push_back("Pixel/" + fedFolderName + "/FEDFatal");
  fatal.push_back("RPC/" + fedFolderName + "/FEDFatal");
  fatal.push_back("SiStrip/" + fedFolderName + "/FEDFatal");

  int k = 0, count = 0;

  float sum = 0.;

  auto ent = entries.begin();
  for (auto fat = fatal.begin(); fat != fatal.end(); ++fat) {
    if (!(dbe_->get(*fat))) {
      reportSummaryContent[k]->Fill(-1);
      reportSummaryMap->setBinContent(1, nSubsystems - k, -1);
      k++;
      ent++;
      continue;
    }

    MonitorElement* me = dbe_->get(*fat);
    MonitorElement* meNorm = dbe_->get(*ent);

    float entry = 0.;
    float norm = 0.;

    if (TH1F* rootHisto = me->getTH1F()) {
      if (TH1F* rootHistoNorm = meNorm->getTH1F()) {
        int Nbins = me->getNbinsX();
        int xmin = (int)rootHisto->GetXaxis()->GetXmin();
        if (*fat == "L1T/" + fedFolderName + "/FEDFatal")
          xmin = xmin + 800;
        if (*fat == "DT/" + fedFolderName + "/FEDFatal")
          xmin = 770;  //Real DT FED IDs are 1369-1371

        float binentry = 0.;
        for (int bin = 1; bin <= Nbins; ++bin) {
          int id = xmin + bin;
          binentry = rootHisto->GetBinContent(bin);
          entry += binentry;
          norm += rootHistoNorm->GetBinContent(bin);
          FedFatal->setBinContent(id, binentry);
        }
      }
    }

    if (norm > 0)
      SummaryContent[k] = 1.0 - entry / norm;
    reportSummaryContent[k]->Fill(SummaryContent[k]);
    if ((k == 2 || k == 3)  // for EE and EB only show yellow when more than 1% errors.
        && SummaryContent[k] >= 0.95 && SummaryContent[k] < 0.99)
      SummaryContent[k] = 0.949;
    reportSummaryMap->setBinContent(1, nSubsystems - k, SummaryContent[k]);
    sum = sum + SummaryContent[k];

    k++;
    ent++;
    count++;
  }

  if (count > 0)
    reportSummary->Fill(sum / (float)count);

  // FED Non Fatal

  std::vector<std::string> nonfatal;
  nonfatal.push_back("CSC/" + fedFolderName + "/FEDNonFatal");
  nonfatal.push_back("DT/" + fedFolderName + "/FEDNonFatal");
  nonfatal.push_back("EcalBarrel/" + fedFolderName + "/FEDNonFatal");
  nonfatal.push_back("EcalEndcap/" + fedFolderName + "/FEDNonFatal");
  nonfatal.push_back("EcalPreshower/" + fedFolderName + "/FEDNonFatal");
  nonfatal.push_back("Hcal/" + fedFolderName + "/FEDNonFatal");
  nonfatal.push_back("L1T/" + fedFolderName + "/FEDNonFatal");
  nonfatal.push_back("Pixel/" + fedFolderName + "/FEDNonFatal");
  nonfatal.push_back("RPC/" + fedFolderName + "/FEDNonFatal");
  nonfatal.push_back("SiStrip/" + fedFolderName + "/FEDNonFatal");

  for (auto non = nonfatal.begin(); non != nonfatal.end(); ++non) {
    if (!(dbe_->get(*non))) {
      continue;
    }

    MonitorElement* me = dbe_->get(*non);

    if (TH1F* rootHisto = me->getTH1F()) {
      int Nbins = me->getNbinsX();
      float entry = 0.;
      int xmin = (int)rootHisto->GetXaxis()->GetXmin();
      if (*non == "L1T/" + fedFolderName + "/FEDNonFatal")
        xmin = xmin + 800;

      for (int bin = 1; bin <= Nbins; ++bin) {
        int id = xmin + bin;
        entry = rootHisto->GetBinContent(bin);
        if (entry > 0.)
          FedNonFatal->setBinContent(id, entry);
      }
    }
  }
}

void DQMFEDIntegrityClient::endRun(const edm::Run& r, const edm::EventSetup& context) {
  if (fillOnEndRun)
    fillHistograms();
}

void DQMFEDIntegrityClient::endJob() {
  if (fillOnEndJob)
    fillHistograms();
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMFEDIntegrityClient);
