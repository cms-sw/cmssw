
/*
 * \file DQMFEDIntegrityClient.cc
 * \author M. Marienfeld
 * Last Update:
 * $Date: 2010/03/29 18:34:06 $
 * $Revision: 1.19 $
 * $Author: ameyer $
 *
 * Description: Summing up FED entries from all subdetectors.
 *
*/

#include "DQMServices/Components/src/DQMFEDIntegrityClient.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// -----------------------------
//  constructors and destructor
// -----------------------------

DQMFEDIntegrityClient::DQMFEDIntegrityClient( const edm::ParameterSet& ps ) {

  parameters_ = ps;
  initialize();
  fillInEventloop = ps.getUntrackedParameter<bool>("fillInEventloop",false);
  fillOnEndRun = ps.getUntrackedParameter<bool>("fillOnEndRun",false);
  fillOnEndJob = ps.getUntrackedParameter<bool>("fillOnEndJob",false);
  fillOnEndLumi = ps.getUntrackedParameter<bool>("fillOnEndLumi",true);
  moduleName = ps.getUntrackedParameter<std::string>("moduleName", "FED");
  fedFolderName = ps.getUntrackedParameter<std::string>("fedFolderName", "FEDIntegrity");

}

DQMFEDIntegrityClient::~DQMFEDIntegrityClient() {

}


void DQMFEDIntegrityClient::initialize() {

  // get back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();

}


void DQMFEDIntegrityClient::beginJob() {

  NBINS = 850;
  XMIN  =   0.;
  XMAX  = 850.;

  dbe_ = edm::Service<DQMStore>().operator->();

  // ----------------------------------------------------------------------------------
  std::string currentFolder = moduleName + "/" + fedFolderName ;
  dbe_->setCurrentFolder(currentFolder.c_str());

  FedEntries  = dbe_->book1D("FedEntries",  "FED Entries",          NBINS, XMIN, XMAX);
  FedFatal    = dbe_->book1D("FedFatal",    "FED Fatal Errors",     NBINS, XMIN, XMAX);
  FedNonFatal = dbe_->book1D("FedNonFatal", "FED Non Fatal Errors", NBINS, XMIN, XMAX);

  FedEntries->setAxisTitle( "", 1);
  FedFatal->setAxisTitle(   "", 1);
  FedNonFatal->setAxisTitle("", 1);

  FedEntries->setAxisTitle( "", 2);
  FedFatal->setAxisTitle(   "", 2);
  FedNonFatal->setAxisTitle("", 2);

  FedEntries->setBinLabel(11,  "PIXEL", 1);
  FedEntries->setBinLabel(221, "SIST",  1);
  FedEntries->setBinLabel(606, "EE",    1);
  FedEntries->setBinLabel(628, "EB",    1);
  FedEntries->setBinLabel(651, "EE",    1);
  FedEntries->setBinLabel(550, "ES",    1);
  FedEntries->setBinLabel(716, "HCAL",  1);
  FedEntries->setBinLabel(754, "CSC",   1);
  FedEntries->setBinLabel(772, "DT",    1);
  FedEntries->setBinLabel(791, "RPC",   1);
  FedEntries->setBinLabel(804, "L1T",   1);

  FedFatal->setBinLabel(11,  "PIXEL", 1);
  FedFatal->setBinLabel(221, "SIST",  1);
  FedFatal->setBinLabel(606, "EE",    1);
  FedFatal->setBinLabel(628, "EB",    1);
  FedFatal->setBinLabel(651, "EE",    1);
  FedFatal->setBinLabel(550, "ES",    1);
  FedFatal->setBinLabel(716, "HCAL",  1);
  FedFatal->setBinLabel(754, "CSC",   1);
  FedFatal->setBinLabel(772, "DT",    1);
  FedFatal->setBinLabel(791, "RPC",   1);
  FedFatal->setBinLabel(804, "L1T",   1);

  FedNonFatal->setBinLabel(11,  "PIXEL", 1);
  FedNonFatal->setBinLabel(221, "SIST",  1);
  FedNonFatal->setBinLabel(606, "EE",    1);
  FedNonFatal->setBinLabel(628, "EB",    1);
  FedNonFatal->setBinLabel(651, "EE",    1);
  FedNonFatal->setBinLabel(550, "ES",    1);
  FedNonFatal->setBinLabel(716, "HCAL",  1);
  FedNonFatal->setBinLabel(754, "CSC",   1);
  FedNonFatal->setBinLabel(772, "DT",    1);
  FedNonFatal->setBinLabel(791, "RPC",   1);
  FedNonFatal->setBinLabel(804, "L1T",   1);

  //-----------------------------------------------------------------------------------
  currentFolder = moduleName + "/EventInfo";
  dbe_->setCurrentFolder(currentFolder.c_str());

  reportSummary = dbe_->bookFloat("reportSummary");

  int nSubsystems = 10;

  if(reportSummary) reportSummary->Fill(1.);

  currentFolder = moduleName + "/EventInfo/reportSummaryContents";
  dbe_->setCurrentFolder(currentFolder.c_str());

  reportSummaryContent[0]  = dbe_->bookFloat("CSC FEDs");
  reportSummaryContent[1]  = dbe_->bookFloat("DT FEDs");
  reportSummaryContent[2]  = dbe_->bookFloat("EB FEDs");
  reportSummaryContent[3]  = dbe_->bookFloat("EE FEDs");
  reportSummaryContent[4]  = dbe_->bookFloat("ES FEDs");
  reportSummaryContent[5]  = dbe_->bookFloat("Hcal FEDs");
  reportSummaryContent[6]  = dbe_->bookFloat("L1T FEDs");
  reportSummaryContent[7]  = dbe_->bookFloat("Pixel FEDs");
  reportSummaryContent[8]  = dbe_->bookFloat("RPC FEDs");
  reportSummaryContent[9] = dbe_->bookFloat("SiStrip FEDs");

  // initialize reportSummaryContents to 1
  for (int i = 0; i < nSubsystems; ++i) {
    SummaryContent[i] = 1.;
    reportSummaryContent[i]->Fill(1.);
  }

  currentFolder = moduleName + "/EventInfo";
  dbe_->setCurrentFolder(currentFolder.c_str());

  reportSummaryMap = dbe_->book2D("reportSummaryMap",
                      "FED Report Summary Map", 1, 1, 2, 10, 1, 11);

  reportSummaryMap->setAxisTitle("", 1);
  reportSummaryMap->setAxisTitle("", 2);

  reportSummaryMap->setBinLabel( 1, " ",       1);
  reportSummaryMap->setBinLabel(10, "CSC",     2);
  reportSummaryMap->setBinLabel( 9, "DT",      2);
  reportSummaryMap->setBinLabel( 8, "EB",      2);
  reportSummaryMap->setBinLabel( 7, "EE",      2);
  reportSummaryMap->setBinLabel( 6, "ES",      2);
  reportSummaryMap->setBinLabel( 5, "Hcal",    2);
  reportSummaryMap->setBinLabel( 4, "L1T",     2);
  reportSummaryMap->setBinLabel( 3, "Pixel",   2);
  reportSummaryMap->setBinLabel( 2, "RPC",     2);
  reportSummaryMap->setBinLabel( 1, "SiStrip", 2);

}

void DQMFEDIntegrityClient::beginRun(const edm::Run& r, const edm::EventSetup& context) {

}

void DQMFEDIntegrityClient::analyze(const edm::Event& e, const edm::EventSetup& context)  {
  if (fillInEventloop) fillHistograms();
}

void DQMFEDIntegrityClient::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& context){
  if (fillOnEndLumi) fillHistograms();
}

void DQMFEDIntegrityClient::fillHistograms(void){
  // FED Entries
  
  // dbe_->showDirStructure();

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

  for(std::vector<std::string>::const_iterator ent = entries.begin();
                                      ent != entries.end(); ++ent) {

    if( !(dbe_->get(*ent)) ) {
      //      cout << ">> Endluminosity No histogram! <<" << endl;
      continue;
    }

    MonitorElement * me = dbe_->get(*ent);

    if (TH1F * rootHisto = me->getTH1F()) {

      int xmin  = 0;
      int Nbins = me->getNbinsX();

      float entry = 0.;

      xmin = (int)rootHisto->GetXaxis()->GetXmin();
      if(*ent == "L1T/" + fedFolderName +"/FEDEntries")  xmin = xmin + 800;

      for(int bin = 1; bin <= Nbins ; ++bin) {
	int id = xmin+bin;
	entry = rootHisto->GetBinContent(bin);
	if(entry > 0.)  FedEntries->setBinContent(id, entry);
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

  std::vector<std::string>::const_iterator ent = entries.begin();
  for(std::vector<std::string>::const_iterator fat = fatal.begin(); 
                                      fat != fatal.end(); ++fat) {

    if( !(dbe_->get(*fat)) ) {
      //      cout << ">> No histogram! <<" << endl;
      reportSummaryContent[k]->Fill(-1);
      reportSummaryMap->setBinContent(1, nSubsystems-k, -1);
      k++;
      ent++;
      continue;
    }

    MonitorElement * me = dbe_->get(*fat);
    MonitorElement * meNorm = dbe_->get(*ent);
      //      cout << "Path : " << me->getFullname() << endl;

    int Nbins  = me->getNbinsX();

    float entry = 0.;
    float norm = 0.;

    if (TH1F * rootHisto = me->getTH1F()) {
      if (TH1F * rootHistoNorm = meNorm->getTH1F()) {

        int xmin   = 0;
        int xmax   = 0;

        xmin = (int)rootHisto->GetXaxis()->GetXmin();
        if(*fat == "L1T/" + fedFolderName + "/FEDFatal") xmin = xmin + 800;

        xmax = (int)rootHisto->GetXaxis()->GetXmax();
        if(*fat == "L1T/" + fedFolderName + "/FEDFatal") xmax = xmax + 800;

        //      cout << "FED ID range : " << xmin << " - " << xmax << endl;

        for(int bin = 1; bin <= Nbins ; ++bin) {
          int id = xmin+bin;
          entry += rootHisto->GetBinContent(bin);
          norm += rootHistoNorm->GetBinContent(bin);
          //      cout << *fat << "errors = " << entry << "\tnorm = " << norm << endl;
          //      cout << "Bin content : " << entry << endl;
          if(entry > 0.) FedFatal->setBinContent(id, entry);
        }

      }
    }

    if (norm > 0) SummaryContent[k] = 1.0 - entry/norm;
    //      cout << "Summary Content : " << SummaryContent[k] << endl;
    reportSummaryContent[k]->Fill(SummaryContent[k]);
    float threshold = 1.;
    if (k==2 || k==3)          // for EE and EB only show yellow when more than 1% errors.
         threshold = 0.99;
    if (SummaryContent[k] < threshold && SummaryContent[k] >=0.95) 
         SummaryContent[k] = 0.949;
    reportSummaryMap->setBinContent(1, nSubsystems-k, SummaryContent[k]);
    sum = sum + SummaryContent[k];

    k++;
    ent++;
    count++;

  }

  if (count > 0) reportSummary->Fill( sum/(float)count );

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

  for(std::vector<std::string>::const_iterator non = nonfatal.begin(); 
                                      non != nonfatal.end(); ++non) {

    if( !(dbe_->get(*non)) ) {
      //      cout << ">> No histogram! <<" << endl;
      continue;
    }

    MonitorElement * me = dbe_->get(*non);

    if (TH1F * rootHisto = me->getTH1F()) {

      int xmin  = 0;
      int Nbins = me->getNbinsX();

      float entry = 0.;

      xmin = (int)rootHisto->GetXaxis()->GetXmin();
      if(*non == "L1T/" + fedFolderName + "/FEDNonFatal") xmin = xmin + 800;

      for(int bin = 1; bin <= Nbins ; ++bin) {
	int id = xmin+bin;
	entry = rootHisto->GetBinContent(bin);
	if(entry > 0.) 	FedNonFatal->setBinContent(id, entry);
      }

    }

  }

}


void DQMFEDIntegrityClient::endRun(const edm::Run& r, const edm::EventSetup& context) {
  if (fillOnEndRun) fillHistograms();
}


void DQMFEDIntegrityClient::endJob() {
  if (fillOnEndJob) fillHistograms();

}
