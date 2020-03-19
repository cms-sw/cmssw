#include "DQMMessageLoggerClient.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <string>
#include <vector>

using namespace std;
using namespace edm;

// -----------------------------
//  constructors and destructor
// -----------------------------

DQMMessageLoggerClient::DQMMessageLoggerClient(const edm::ParameterSet& ps) {
  parameters = ps;
  theDbe = nullptr;
  modulesErrorsFound = nullptr;
  modulesWarningsFound = nullptr;
  categoriesWarningsFound = nullptr;
  categoriesErrorsFound = nullptr;
  directoryName = parameters.getParameter<string>("Directory");
}

DQMMessageLoggerClient::~DQMMessageLoggerClient() = default;

void DQMMessageLoggerClient::beginJob() {
  //LogTrace(metname)<<"[DQMMessageLoggerClient] Parameters initialization";
  theDbe = Service<DQMStore>().operator->();

  if (theDbe != nullptr) {
    theDbe->setCurrentFolder(directoryName);
  }
}

void DQMMessageLoggerClient::beginRun(const edm::Run& r, const edm::EventSetup& es) {}

void DQMMessageLoggerClient::analyze(const edm::Event& e, const edm::EventSetup& context) {}

void DQMMessageLoggerClient::fillHistograms() {
  // directoryName should be the same as for DQMMessageLogger
  //theDbe->setCurrentFolder(directoryName);
  /*
  cout << theDbe->pwd() << endl;
  vector<string> vec = theDbe->getSubdirs();
  for(int i=0; i<vec.size(); i++){
    cout << vec[i] << endl;
  }
  */
  theDbe->goUp();

  vector<string> entries;
  entries.push_back(directoryName + "/Warnings/modules_warnings");
  entries.push_back(directoryName + "/Errors/modules_errors");
  entries.push_back(directoryName + "/Warnings/categories_warnings");
  entries.push_back(directoryName + "/Errors/categories_errors");

  int mel = 0;

  for (auto ent = entries.begin(); ent != entries.end(); ++ent) {
    mel++;
    //RESET VECTORS
    binContent.clear();
    binLabel.clear();

    // RETURN ME

    MonitorElement* me = theDbe->get(*ent);
    // GET TH1F
    if (theDbe->get(*ent)) {
      if (TH1* rootHisto = me->getTH1()) {
        int nonzeros = 0;
        int Nbins = me->getNbinsX();

        // LOOP OVER TH1F
        for (int bin = 1; bin <= Nbins; ++bin) {
          if (rootHisto->GetBinContent(bin) > 0) {
            nonzeros++;
            binContent.push_back(rootHisto->GetBinContent(bin));
            binLabel.emplace_back(rootHisto->GetXaxis()->GetBinLabel(bin));
          }
        }

        switch (mel) {
          case 1:
            theDbe->setCurrentFolder(directoryName + "/Warnings");
            modulesWarningsFound = theDbe->get(directoryName + "/Warnings/modulesWarningsFound");
            if (nonzeros > 0) {
              modulesWarningsFound = theDbe->book1D(
                  "modulesWarningsFound", "Warnings per module", binContent.size(), 0, binContent.size());
            } else {
              modulesWarningsFound = theDbe->book1D("modulesWarningsFound", "Warnings per module", 1, 0, 1);
              modulesWarningsFound->setBinLabel(1, "Module name");
            }
            for (int i = 0; i < nonzeros; ++i) {
              if (modulesWarningsFound != nullptr) {
                //gPad->SetBottomMargin(2);
                //cout << binContent[i] <<" "<<binLabel[i] << endl;
                modulesWarningsFound->setBinContent(i + 1, binContent[i]);
                modulesWarningsFound->setBinLabel(i + 1, binLabel[i]);
              }
            }
            if (nonzeros > 4)
              modulesWarningsFound->getTH1()->GetXaxis()->LabelsOption("v");
            break;
          case 2:
            theDbe->setCurrentFolder(directoryName + "/Errors");
            modulesErrorsFound = theDbe->get(directoryName + "/Errors/modulesErrorsFound");
            if (nonzeros > 0) {
              modulesErrorsFound =
                  theDbe->book1D("modulesErrorsFound", "Errors per module", binContent.size(), 0, binContent.size());
            } else {
              modulesErrorsFound = theDbe->book1D("modulesErrorsFound", "Errors per module", 1, 0, 1);
              modulesErrorsFound->setBinLabel(1, "Module name");
            }
            for (int i = 0; i < nonzeros; ++i) {
              if (modulesErrorsFound != nullptr) {
                //gPad->SetBottomMargin(2);
                modulesErrorsFound->setBinContent(i + 1, binContent[i]);
                modulesErrorsFound->setBinLabel(i + 1, binLabel[i]);
              }
            }
            if (nonzeros > 4)
              modulesErrorsFound->getTH1()->GetXaxis()->LabelsOption("v");
            break;
          case 3:
            theDbe->setCurrentFolder(directoryName + "/Warnings");
            categoriesWarningsFound = theDbe->get(directoryName + "/Warnings/categoriesWarningsFound");
            if (nonzeros > 0) {
              categoriesWarningsFound = theDbe->book1D(
                  "categoriesWarningsFound", "Warnings per category", binContent.size(), 0, binContent.size());
            } else {
              categoriesWarningsFound = theDbe->book1D("categoriesWarningsFound", "Warnings per category", 1, 0, 1);
              categoriesWarningsFound->setBinLabel(1, "Category name");
            }
            for (int i = 0; i < nonzeros; ++i) {
              if (categoriesWarningsFound != nullptr) {
                //gPad->SetBottomMargin(2);
                //cout << binContent[i] <<" " <<binLabel[i] << endl;
                categoriesWarningsFound->setBinContent(i + 1, binContent[i]);
                categoriesWarningsFound->setBinLabel(i + 1, binLabel[i]);
              }
            }
            if (nonzeros > 4)
              categoriesWarningsFound->getTH1()->GetXaxis()->LabelsOption("v");
            break;
          case 4:
            theDbe->setCurrentFolder(directoryName + "/Errors");
            categoriesErrorsFound = theDbe->get(directoryName + "/Errors/categoriesErrorsFound");
            if (nonzeros > 0) {
              categoriesErrorsFound = theDbe->book1D(
                  "categoriesErrorsFound", "Errors per category", binContent.size(), 0, binContent.size());
            } else {
              categoriesErrorsFound = theDbe->book1D("categoriesErrorsFound", "Errors per category", 1, 0, 1);
              categoriesErrorsFound->setBinLabel(1, "Category name");
            }
            for (int i = 0; i < nonzeros; ++i) {
              if (categoriesErrorsFound != nullptr) {
                //gPad->SetBottomMargin(2);
                categoriesErrorsFound->setBinContent(i + 1, binContent[i]);
                categoriesErrorsFound->setBinLabel(i + 1, binLabel[i]);
              }
            }
            if (nonzeros > 4)
              categoriesErrorsFound->getTH1()->GetXaxis()->LabelsOption("v");
            break;
        }
      }
    }
  }
}

void DQMMessageLoggerClient::endRun(const Run& r, const EventSetup& es) { fillHistograms(); }

void DQMMessageLoggerClient::endJob() {
  //LogTrace(metname)<<"[DQMMessageLoggerClient] EndJob";
}
