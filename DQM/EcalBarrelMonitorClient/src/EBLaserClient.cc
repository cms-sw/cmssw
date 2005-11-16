/*
 * \file EBLaserClient.cc
 * 
 * $Date: 2005/11/16 08:36:44 $
 * $Revision: 1.15 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBLaserClient.h>

EBLaserClient::EBLaserClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  Char_t histo[50];

  for ( int i = 0; i < 36; i++ ) {

    h01_[i] = 0;
    h02_[i] = 0;
    h03_[i] = 0;
    h04_[i] = 0;

  }

}

EBLaserClient::~EBLaserClient(){

  this->unsubscribe();

  for ( int i = 0; i < 36; i++ ) {

    if ( h01_[i] ) delete h01_[i];
    if ( h02_[i] ) delete h02_[i];
    if ( h03_[i] ) delete h03_[i];
    if ( h04_[i] ) delete h04_[i];

  }

}

void EBLaserClient::beginJob(const edm::EventSetup& c){

  cout << "EBLaserClient: beginJob" << endl;

  ievt_ = 0;

}

void EBLaserClient::beginRun(const edm::EventSetup& c){

  cout << "EBLaserClient: beginRun" << endl;

  jevt_ = 0;

  this->subscribe();

  for ( int i = 0; i < 36; i++ ) {

    if ( h01_[i] ) delete h01_[i];
    if ( h02_[i] ) delete h02_[i];
    if ( h03_[i] ) delete h03_[i];
    if ( h04_[i] ) delete h04_[i];
    h01_[i] = 0;
    h02_[i] = 0;
    h03_[i] = 0;
    h04_[i] = 0;

  }

}

void EBLaserClient::endJob(void) {

  cout << "EBLaserClient: endJob, ievt = " << ievt_ << endl;

}

void EBLaserClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBLaserClient: endRun, jevt = " << jevt_ << endl;

  if ( jevt_ == 0 ) return;

  EcalLogicID ecid;
  MonLaserBlueDat apdb;
  map<EcalLogicID, MonLaserBlueDat> datasetb;
  MonLaserGreenDat apdg;
  map<EcalLogicID, MonLaserGreenDat> datasetg;
  MonLaserInfraredDat apdi;
  map<EcalLogicID, MonLaserInfraredDat> dataseti;
  MonLaserRedDat apdr;
  map<EcalLogicID, MonLaserRedDat> datasetr;

  cout << "Writing MonLaserDatObjects to database ..." << endl;

  float n_min_tot = 1000.;
  float n_min_bin = 50.;
  
  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03, num04;
    float mean01, mean02, mean03, mean04;
    float rms01, rms02, rms03, rms04;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = num03  = num04  = -1.;
        mean01 = mean02 = mean03 = mean04 = -1.;
        rms01  = rms02  = rms03  = rms04  = -1.;

        bool update_channel = false;

        if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
            rms01  = h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( h02_[ism-1] && h02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = h02_[ism-1]->GetBinEntries(h02_[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = h02_[ism-1]->GetBinContent(h02_[ism-1]->GetBin(ie, ip));
            rms02  = h02_[ism-1]->GetBinError(h02_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( h03_[ism-1] && h03_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = h03_[ism-1]->GetBinEntries(h03_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            mean03 = h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(ie, ip));
            rms03  = h03_[ism-1]->GetBinError(h03_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( h04_[ism-1] && h04_[ism-1]->GetEntries() >= n_min_tot ) {
          num04 = h04_[ism-1]->GetBinEntries(h04_[ism-1]->GetBin(ie, ip));
          if ( num04 >= n_min_bin ) {
            mean04 = h04_[ism-1]->GetBinContent(h04_[ism-1]->GetBin(ie, ip));
            rms04  = h04_[ism-1]->GetBinError(h04_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( update_channel ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Inserting dataset for SM=" << ism << endl;

            cout << "L1 (" << ie << "," << ip << ") " << num01 << " " << mean01 << " " << rms01 << endl;
            cout << "L2 (" << ie << "," << ip << ") " << num03 << " " << mean03 << " " << rms03 << endl;

          }

          apdb.setAPDMean(mean01);
          apdb.setAPDRMS(rms01);

          apdb.setAPDOverPNMean(mean02);
          apdb.setAPDOverPNRMS(rms02);

          apdb.setTaskStatus(1);

          apdr.setAPDMean(mean03);
          apdr.setAPDRMS(rms03);

          apdr.setAPDOverPNMean(mean04);
          apdr.setAPDOverPNRMS(rms04);

          apdr.setTaskStatus(1);

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
              datasetb[ecid] = apdb;
              datasetr[ecid] = apdr;
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
      econn->insertDataSet(&datasetb, runiov, runtag );
      econn->insertDataSet(&datasetr, runiov, runtag );
      cout << "done." << endl; 
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBLaserClient::subscribe(void){

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM*");

}

void EBLaserClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM*");

}

void EBLaserClient::unsubscribe(void){

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM*");

}

void EBLaserClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 )  
    cout << "EBLaserClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  this->subscribeNew();

  Char_t histo[150];
  
  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    sprintf(histo, "Collector/FU0/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h01_[ism-1] ) delete h01_[ism-1];
        h01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h02_[ism-1] ) delete h02_[ism-1];
        h02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      } 
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h03_[ism-1] ) delete h03_[ism-1];
        h03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      } 
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h04_[ism-1] ) delete h04_[ism-1];
        h04_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      } 
    }

  }

}

void EBLaserClient::htmlOutput(int run, string htmlDir){

  cout << "Preparing EBLaserClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + "EBLaserClient.html").c_str());


  htmlFile.close();

}

