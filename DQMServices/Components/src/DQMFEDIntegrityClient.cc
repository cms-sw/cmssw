
/*
 * \file DQMFEDIntegrityClient.cc
 * \author M. Marienfeld
 * Last Update:
 * $Date: 2009/06/03 09:42:07 $
 * $Revision: 1.6 $
 * $Author: ameyer $
 *
 * Description: Summing up FED entries from all subdetectors.
 *
*/

#include "DQMServices/Components/src/DQMFEDIntegrityClient.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TRandom.h"
#include <math.h>

using namespace std;
using namespace edm;


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
  moduleName = ps.getUntrackedParameter<string>("moduleName", "FED") ;

}


DQMFEDIntegrityClient::~DQMFEDIntegrityClient() {

}


void DQMFEDIntegrityClient::initialize() {

  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();

}


void DQMFEDIntegrityClient::beginJob(const EventSetup& context) {

  NBINS = 850;
  XMIN  =   0.;
  XMAX  = 850.;

  dbe_ = Service<DQMStore>().operator->();

  // ----------------------------------------------------------------------------------
  string subFolder = "/FEDIntegrity";
  string currentFolder = moduleName + subFolder;

  //  dbe_->setCurrentFolder("FED/FEDIntegrity");
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
  subFolder = "/EventInfo";
  currentFolder = moduleName + subFolder;

  //  dbe_->setCurrentFolder("FED/EventInfo");
  dbe_->setCurrentFolder(currentFolder.c_str());

  reportSummary = dbe_->bookFloat("reportSummary");

  int nSubsystems = 10;

  if(reportSummary) reportSummary->Fill(1.);

  subFolder = "/EventInfo/reportSummaryContents";
  currentFolder = moduleName + subFolder;

  //  dbe_->setCurrentFolder("FED/EventInfo/reportSummaryContents");
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

  subFolder = "/EventInfo";
  currentFolder = moduleName + subFolder;

  //  dbe_->setCurrentFolder("FED/EventInfo");
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

void DQMFEDIntegrityClient::beginRun(const edm::Run& r, const EventSetup& context) {

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

  vector<string> entries;
  entries.push_back("CSC/FEDIntegrity/FEDEntries");
  entries.push_back("DT/FEDIntegrity/FEDEntries");
  entries.push_back("EcalBarrel/FEDIntegrity/FEDEntries");
  entries.push_back("EcalEndcap/FEDIntegrity/FEDEntries");
  entries.push_back("EcalPreshower/FEDIntegrity/FEDEntries");
  entries.push_back("Hcal/FEDIntegrity/FEDEntries");
  entries.push_back("L1T/FEDIntegrity/FEDEntries");
  entries.push_back("Pixel/FEDIntegrity/FEDEntries");
  entries.push_back("RPC/FEDIntegrity/FEDEntries");
  entries.push_back("SiStrip/FEDIntegrity/FEDEntries");

  for(vector<string>::const_iterator ent = entries.begin();
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
      if(*ent == "L1T/FEDIntegrity/FEDEntries")  xmin = xmin + 800;

      for(int bin = 1; bin <= Nbins ; ++bin) {
	int id = xmin+bin;
	entry = rootHisto->GetBinContent(bin);
	if(entry > 0.)  FedEntries->setBinContent(id, entry);
      }

    }

  }

  // FED Fatal

  int nSubsystems = 10;

  vector<string> fatal;
  fatal.push_back("CSC/FEDIntegrity/FEDFatal");
  fatal.push_back("DT/FEDIntegrity/FEDFatal");
  fatal.push_back("EcalBarrel/FEDIntegrity/FEDFatal");
  fatal.push_back("EcalEndcap/FEDIntegrity/FEDFatal");
  fatal.push_back("EcalPreshower/FEDIntegrity/FEDFatal");
  fatal.push_back("Hcal/FEDIntegrity/FEDFatal");
  fatal.push_back("L1T/FEDIntegrity/FEDFatal");
  fatal.push_back("Pixel/FEDIntegrity/FEDFatal");
  fatal.push_back("RPC/FEDIntegrity/FEDFatal");
  fatal.push_back("SiStrip/FEDIntegrity/FEDFatal");

  int k = 0, count = 0;

  float sum = 0.;

  for(vector<string>::const_iterator fat = fatal.begin(); 
                                      fat != fatal.end(); ++fat) {

    if( !(dbe_->get(*fat)) ) {
      //      cout << ">> No histogram! <<" << endl;
      reportSummaryContent[k]->Fill(-1);
      reportSummaryMap->setBinContent(1, nSubsystems-k, -1);
      k++;
      continue;
    }

    MonitorElement * me = dbe_->get(*fat);
      //      cout << "Path : " << me->getFullname() << endl;

    int Nfatal = 0;
    int Nbins  = me->getNbinsX();

    if (TH1F * rootHisto = me->getTH1F()) {

      int xmin   = 0;
      int xmax   = 0;

      float entry = 0.;

      xmin = (int)rootHisto->GetXaxis()->GetXmin();
      if(*fat == "L1T/FEDIntegrity/FEDFatal") xmin = xmin + 800;

      xmax = (int)rootHisto->GetXaxis()->GetXmax();
      if(*fat == "L1T/FEDIntegrity/FEDFatal") xmax = xmax + 800;

      //      cout << "FED ID range : " << xmin << " - " << xmax << endl;

      for(int bin = 1; bin <= Nbins ; ++bin) {
	int id = xmin+bin;
	entry = rootHisto->GetBinContent(bin);
      //      cout << "Bin content : " << entry << endl;
	if(entry > 0.) {
	  ++Nfatal;
	  FedFatal->setBinContent(id, entry);
	}
      }

    }

    if(Nbins > 0) SummaryContent[k] = 1.-((float)Nfatal/(float)Nbins);
      //      cout << "Summary Content : " << SummaryContent[k] << endl;
    reportSummaryContent[k]->Fill(SummaryContent[k]);
    reportSummaryMap->setBinContent(1, nSubsystems-k, SummaryContent[k]);
    sum = sum + SummaryContent[k];

    k++;
    count++;

  }

  if(count > 0)  reportSummary->Fill( sum/(float)count );

  // FED Non Fatal

  vector<string> nonfatal;
  nonfatal.push_back("CSC/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("DT/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("EcalBarrel/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("EcalEndcap/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("EcalPreshower/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("Hcal/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("L1T/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("Pixel/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("RPC/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("SiStrip/FEDIntegrity/FEDNonFatal");

  for(vector<string>::const_iterator non = nonfatal.begin(); 
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
      if(*non == "L1T/FEDIntegrity/FEDNonFatal") xmin = xmin + 800;

      for(int bin = 1; bin <= Nbins ; ++bin) {
	int id = xmin+bin;
	entry = rootHisto->GetBinContent(bin);
	if(entry > 0.) 	FedNonFatal->setBinContent(id, entry);
      }

    }

  }

}


void DQMFEDIntegrityClient::endRun(const Run& r, const EventSetup& context) {
  if (fillOnEndRun) fillHistograms();
}


void DQMFEDIntegrityClient::endJob() {
  if (fillOnEndJob) fillHistograms();

}
