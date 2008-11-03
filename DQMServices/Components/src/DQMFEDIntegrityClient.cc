
/*
 * \file DQMFEDIntegrityClient.cc
 * \author M. Marienfeld
 * Last Update:
 * $Date: 2008/11/03 15:26:25 $
 * $Revision: 1.1 $
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

}


DQMFEDIntegrityClient::~DQMFEDIntegrityClient() {

}


void DQMFEDIntegrityClient::initialize() {

  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();

}


void DQMFEDIntegrityClient::beginJob(const EventSetup& context) {

  NBINS = 820;
  XMIN  =    0.;
  XMAX  = 820.;

  dbe_ = Service<DQMStore>().operator->();

  dbe_->setCurrentFolder("FED/FEDIntegrity");

  FedEntries  = dbe_->book1D("FedEntries",  "FED Entries",          NBINS, XMIN, XMAX);
  FedFatal    = dbe_->book1D("FedFatal",    "FED Fatal Errors",     NBINS, XMIN, XMAX);
  FedNonFatal = dbe_->book1D("FedNonFatal", "FED Non Fatal Errors", NBINS, XMIN, XMAX);

  dbe_->setCurrentFolder("FED/EventInfo");
  
  reportSummary = dbe_->bookFloat("reportSummary");

  int nSubsystems = 7;

  // initialize reportSummary to 1
  if(reportSummary) reportSummary->Fill(1.);

  dbe_->setCurrentFolder("FED/EventInfo/reportSummaryContents");

  reportSummaryContent[0] = dbe_->bookFloat("CSC FEDs");
  reportSummaryContent[1] = dbe_->bookFloat("DT FEDs");
  reportSummaryContent[2] = dbe_->bookFloat("EB FEDs");
  reportSummaryContent[3] = dbe_->bookFloat("EE FEDs");
  reportSummaryContent[4] = dbe_->bookFloat("L1T FEDs");
  reportSummaryContent[5] = dbe_->bookFloat("Pixel FEDs");
  reportSummaryContent[6] = dbe_->bookFloat("Strip FEDs");

  // initialize reportSummaryContents to 1
  for (int i = 0; i < nSubsystems; ++i) {
    SummaryContent[i] = 1.;
    reportSummaryContent[i]->Fill(1.);
  }

  dbe_->setCurrentFolder("FED/EventInfo");

  reportSummaryMap = dbe_->book2D("reportSummaryMap", "FED Report Summary Map", 1, 1, 2, 7, 1, 8);
  reportSummaryMap->setAxisTitle("", 1);
  reportSummaryMap->setAxisTitle("", 2);
  reportSummaryMap->setBinLabel(1, "CSC", 2);
  reportSummaryMap->setBinLabel(2, "DT", 2);
  reportSummaryMap->setBinLabel(3, "EB", 2);
  reportSummaryMap->setBinLabel(4, "EE", 2);
  reportSummaryMap->setBinLabel(5, "L1T", 2);
  reportSummaryMap->setBinLabel(6, "Pixel", 2);
  reportSummaryMap->setBinLabel(7, "Strip", 2);
  reportSummaryMap->setBinLabel(1, " ", 1);

}


void DQMFEDIntegrityClient::beginRun(const edm::Run& r, const EventSetup& context) {

}


void DQMFEDIntegrityClient::analyze(const Event& iEvent, const EventSetup& iSetup) {

  // FED Entries

  vector<string> entries;
  entries.push_back("CSC/FEDIntegrity/FEDEntries");
  entries.push_back("DT/FEDIntegrity_EvF/FEDEntries");
  //  entries.push_back("DT/FEDIntegrity/FEDEntries");
  entries.push_back("EcalBarrel/FEDIntegrity/FEDEntries");
  entries.push_back("EcalEndcap/FEDIntegrity/FEDEntries");
//  entries.push_back("L1T/FEDIntegrity/FEDEntries");
  entries.push_back("Pixel/FEDIntegrity/FEDEntries");
  entries.push_back("SiStrip/FEDIntegrity/FEDEntries");

  for(vector<string>::const_iterator ent = entries.begin();
                                      ent != entries.end(); ++ent) {

    MonitorElement * me = dbe_->get(*ent);

    if (TH1F * rootHisto = me->getTH1F()) {

      int xmin  = 0;
      int xmax  = 0;
      int Nbins = me->getNbinsX();

      float entry = 0.;

      xmin = (int)rootHisto->GetXaxis()->GetXmin();
      xmax = (int)rootHisto->GetXaxis()->GetXmax();

      for(int bin = 1; bin <= Nbins ; ++bin) {
	int id = xmin+bin;
	entry = rootHisto->GetBinContent(bin);
	FedEntries->setBinContent(id, entry);
      }

    }

  }

  // FED Fatal

  int nSubsystems = 7;

  vector<string> fatal;
  fatal.push_back("CSC/FEDIntegrity/FEDFatal");
  fatal.push_back("DT/FEDIntegrity_EvF/FEDFatal");
  //  fatal.push_back("DT/FEDIntegrity/FEDFatal");
  fatal.push_back("EcalBarrel/FEDIntegrity/FEDFatal");
  fatal.push_back("EcalEndcap/FEDIntegrity/FEDFatal");
//  fatal.push_back("L1T/FEDIntegrity/FEDFatal");
  fatal.push_back("Pixel/FEDIntegrity/FEDFatal");
  fatal.push_back("SiStrip/FEDIntegrity/FEDFatal");

  int k=0;
  for(vector<string>::const_iterator fat = fatal.begin(); 
                                      fat != fatal.end(); ++fat) {
    MonitorElement * me = dbe_->get(*fat);
    //cout << "New Module!" << endl;
    //cout << "Path : " << me->getFullname() << endl;

    int Nfatal = 0;
    int Nbins  = me->getNbinsX();

    if (TH1F * rootHisto = me->getTH1F()) {

      int xmin   = 0;
      int xmax   = 0;

      float entry = 0.;

      xmin = (int)rootHisto->GetXaxis()->GetXmin();
      xmax = (int)rootHisto->GetXaxis()->GetXmax();
      //cout << "FED ID range : " << xmin << " - " << xmax << endl;

      for(int bin = 1; bin <= Nbins ; ++bin) {
	int id = xmin+bin;
	//	cout << "FED ID      : " << id << endl;
	entry = rootHisto->GetBinContent(bin);
	//	cout << "Bin content : " << entry << endl;
	if(entry > 0) ++Nfatal;
	FedFatal->setBinContent(id, entry);
      }

    }

    //    cout << "Nfatal : " << Nfatal << endl;
    //    cout << "Nbins  : " << Nbins  << endl;

    if(Nbins > 0) SummaryContent[k] = 1.-((float)Nfatal/(float)Nbins);
    //cout << "Summary Content : " << SummaryContent[k] << endl;
    reportSummaryContent[k]->Fill(SummaryContent[k]);
    reportSummaryMap->setBinContent(1, k+1, SummaryContent[k]);
    k++;
  }

  float sum = 0.;

  for(int i = 0; i < nSubsystems; ++i) {
    sum = sum + SummaryContent[i];
  }

  if(nSubsystems > 0) reportSummary->Fill( sum/(float)nSubsystems );

  // FED Non Fatal

  vector<string> nonfatal;
  nonfatal.push_back("CSC/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("DT/FEDIntegrity_EvF/FEDNonFatal");
  //  nonfatal.push_back("DT/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("EcalBarrel/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("EcalEndcap/FEDIntegrity/FEDNonFatal");
//  nonfatal.push_back("L1T/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("Pixel/FEDIntegrity/FEDNonFatal");
  nonfatal.push_back("SiStrip/FEDIntegrity/FEDNonFatal");

  for(vector<string>::const_iterator non = nonfatal.begin(); 
                                      non != nonfatal.end(); ++non) {

    MonitorElement * me = dbe_->get(*non);

    if (TH1F * rootHisto = me->getTH1F()) {

      int xmin  = 0;
      int xmax  = 0;
      int Nbins = me->getNbinsX();

      float entry = 0.;

      xmin = (int)rootHisto->GetXaxis()->GetXmin();
      xmax = (int)rootHisto->GetXaxis()->GetXmax();

      for(int bin = 1; bin <= Nbins ; ++bin) {
	int id = xmin+bin;
	entry = rootHisto->GetBinContent(bin);
	FedNonFatal->setBinContent(id, entry);
      }

    }

  }

}


void DQMFEDIntegrityClient::endRun(const Run& r, const EventSetup& context) {

}


void DQMFEDIntegrityClient::endJob() {

}
