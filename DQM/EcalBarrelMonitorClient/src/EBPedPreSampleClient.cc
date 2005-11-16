/*
 * \file EBPedPreSampleClient.cc
 * 
 * $Date: 2005/11/16 08:36:44 $
 * $Revision: 1.17 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBPedPreSampleClient.h>

EBPedPreSampleClient::EBPedPreSampleClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  Char_t histo[50];

  for ( int i = 0; i < 36; i++ ) {

    h03_[i] = 0;

    sprintf(histo, "EBPT pedestal PreSample quality G12 SM%02d", i+1);
    g03_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    sprintf(histo, "EBPT pedestal PreSample mean G01 SM%02d", i+1);
    p03_[i] = new TH1F(histo, histo, 100, 150., 250.);

    sprintf(histo, "EBPT pedestal PreSample rms G01 SM%02d", i+1);
    r03_[i] = new TH1F(histo, histo, 100, 0., 10.);

  }

}

EBPedPreSampleClient::~EBPedPreSampleClient(){

  this->unsubscribe();

  for ( int i = 0; i < 36; i++ ) {

    if ( h03_[i] ) delete h03_[i];

    delete g03_[i];

    delete p03_[i];

    delete r03_[i];

  }

}

void EBPedPreSampleClient::beginJob(const edm::EventSetup& c){

  cout << "EBPedPreSampleClient: beginJob" << endl;

  ievt_ = 0;

}

void EBPedPreSampleClient::beginRun(const edm::EventSetup& c){

  cout << "EBPedPreSampleClient: beginRun" << endl;

  jevt_ = 0;

  this->subscribe();

  for ( int i = 0; i < 36; i++ ) {

    if ( h03_[i] ) delete h03_[i];
    h03_[i] = 0;

  }

}

void EBPedPreSampleClient::endJob(void) {

  cout << "EBPedPreSampleClient: endJob, ievt = " << ievt_ << endl;

}

void EBPedPreSampleClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBPedPreSampleClient: endRun, jevt = " << jevt_ << endl;

  if ( jevt_ == 0 ) return;

  EcalLogicID ecid;
//  MonPedestalsDat p;
//  map<EcalLogicID, MonPedestalsDat> dataset;

  cout << "Writing MonPedPreSampleDatObjects to database ..." << endl;

  float n_min_tot = 1000.;
  float n_min_bin = 50.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num03;
    float mean03;
    float rms03;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num03  = -1.;
        mean03 = -1.;
        rms03  = -1.;

        bool update_channel = false;

        if ( h03_[ism-1] && h03_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = h03_[ism-1]->GetBinEntries(h03_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            mean03 = h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(ie, ip));
            rms03  = h03_[ism-1]->GetBinError(h03_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( update_channel ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Inserting dataset for SM=" << ism << endl;

            cout << "G12 (" << ie << "," << ip << ") " << num03  << " "
                                                       << mean03 << " "
                                                       << rms03  << endl;
          }

//          p.setPedMeanG12(mean03);
//          p.setPedRMSG12(rms03);

//          p.setTaskStatus(1);

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
//              dataset[ecid] = p;
            } catch (runtime_error &e) {
              cerr << e.what() << endl;
            }
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      cout << "Inserting dataset ... " << flush;
//      econn->insertDataSet(&dataset, runiov, runtag );
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBPedPreSampleClient::subscribe(void){

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM*");

}

void EBPedPreSampleClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM*");

}

void EBPedPreSampleClient::unsubscribe(void){

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM*");

}

void EBPedPreSampleClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 )  
    cout << "EBPedPreSampleClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  this->subscribeNew();

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    sprintf(histo, "Collector/FU0/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM%02d G12", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h03_[ism-1] ) delete h03_[ism-1];
        h03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      }
    }

  }

}

void EBPedPreSampleClient::htmlOutput(int run, string htmlDir){

  cout << "Preparing EBPedPreSampleClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + "EBPedPreSampleClient.html").c_str());


  htmlFile.close();

}

